// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Python-binding for a CUTLASS GEMM with the SAME "heavy" fused epilogue as
// `heavy_epi_torch_ext.cu`, but built on the **arch::Sm120 dense mainloop**
// (Blackwell-consumer F8F6F4 Tensor Core MMA) plus the **CUTLASS 3.x Sm90EVT**
// epilogue API.
//
// Why FP8 instead of bf16
// -----------------------
// `cutlass::gemm::collective::CollectiveBuilder<arch::Sm120, ...>` enforces
//
//     static_assert("SM120 TmaWarpSpecialized builder currently only supports F8F6F4 MMA.")
//
// (`include/cutlass/gemm/collective/builders/sm120_mma_builder.inl:82`).  There
// is no Sm120 dense bf16 mainloop; bf16 input on this arch lives only on the
// Sm90 WGMMA path (`heavy_epi_90_torch_ext.cu`).  Picking FP8 E4M3 keeps
// quantisation noise small while staying fully on the Sm120 path.
//
// Pairs with
//   * `heavy_epi_torch_ext.cu`        — same epilogue, CUTLASS 2.x Sm80 EVT (bf16 A/B)
//   * `heavy_epi_90_torch_ext.cu`     — same epilogue, CUTLASS 3.x Sm90EVT (bf16 A/B)
//   * `heavy_epi_low_precision_torch_ext.cu` — same epilogue, multi-dtype Sm120 path
//
// The fused computation is the same as the bf16 reference:
//
//     D = tanh( SiLU(A @ B + bias_row) * scale_col + Aux )
//
//     A         : (M, K) FP8 E4M3 RowMajor  (packed uint8, K-major)
//     B         : (K, N) FP8 E4M3 ColMajor  (packed uint8, K-major; TN required)
//     bias_row  : (N,)   bf16              (broadcast along M)
//     scale_col : (M,)   bf16              (broadcast along N)
//     Aux       : (M, N) bf16 RowMajor      (bound to the GEMM "C" operand)
//     D         : (M, N) bf16 RowMajor
//
// Sm90EVT tree (re-used on Sm120 because the Sm120 epilogue accepts the same
// fusion-callbacks API):
//
//     T0 = Acc + Bias            (Sm90AccFetch + Sm90RowBroadcast<Bias>)
//     T1 = SiLU(T0)              (Sm90Compute<SiLu>)
//     T2 = T1 * Scale            (Sm90ColBroadcast<Scale> * _)
//     T3 = T2 + SrcFetch(C=Aux)  (Sm90SrcFetch<C> serves as the Aux load)
//     D  = tanh(T3)              (Sm90Compute<Tanh>, final store)
//
// Accumulation is fp32, epilogue compute is fp32, output is bf16.
//
// Alignment requirements (`detail::get_input_alignment_bits` for FP8):
//   K must be a multiple of 16 (=128 bits / 8 bits-per-element); the 128x128
//   TileShape adds M%128 == 0 and N%128 == 0.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/functional.h"
#include "cutlass/numeric_types.h"
#include "cutlass/float8.h"

#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/util/packed_stride.hpp"

////////////////////////////////////////////////////////////////////////////////

using namespace cute;

namespace heavy_epi_120_90evt {

////////////////////////////////////////////////////////////////////////////////
// Compile-time GEMM configuration.  All knobs (tile shape, EVT tree, schedule)
// are fixed; A/B are FP8 E4M3, epilogue tensors are bf16.
////////////////////////////////////////////////////////////////////////////////

struct GemmConfig {
  // Sm120 dense MMA = F8F6F4.  Pick FP8 E4M3 — the simplest of the family
  // and the smallest quantisation gap relative to bf16.
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;

  using ElementC     = cutlass::bfloat16_t;     // bound to Aux via SrcFetch
  using ElementD     = cutlass::bfloat16_t;
  using ElementBias  = cutlass::bfloat16_t;
  using ElementScale = cutlass::bfloat16_t;

  using ElementAccumulator = float;
  using ElementCompute     = float;

  // 128 bits / 8 bits per E4M3 element = 16 elements
  static constexpr int AlignmentA = 16;
  static constexpr int AlignmentB = 16;
  // 128 bits / 16 bits per bf16 element = 8 elements
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  // sm_120 F8F6F4 builder requires TN: A RowMajor, B ColumnMajor (both K-major).
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;

  using ArchTag       = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using TileShape    = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;           // required on sm_120

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

  //
  // Heavy epilogue EVT (Sm90EVT, accepted by the Sm120 epilogue builder):
  //
  //     tanh( ( SiLU(Acc + Bias) * Scale ) + C[Aux] )
  //
  //   └─ Sm90Compute<Tanh>
  //        └─ Sm90Compute<plus>                           // + Aux (via C)
  //             ├─ Sm90Compute<multiplies>                // * Scale
  //             │    ├─ Sm90Compute<SiLu>                 // SiLU(_)
  //             │    │    └─ Sm90Compute<plus>            // Acc + Bias
  //             │    │         ├─ Sm90AccFetch
  //             │    │         └─ Sm90RowBroadcast<Bias>
  //             │    └─ Sm90ColBroadcast<Scale>
  //             └─ Sm90SrcFetch<C>                        // C serves as Aux
  //
  using BiasLoad = cutlass::epilogue::fusion::Sm90RowBroadcast<
      /*Stages=*/0, TileShape, ElementBias, ElementCompute>;
  using ScaleLoad = cutlass::epilogue::fusion::Sm90ColBroadcast<
      /*Stages=*/0, TileShape, ElementScale, ElementCompute>;
  using AuxLoadViaC = cutlass::epilogue::fusion::Sm90SrcFetch<ElementC>;

  // Acc + Bias
  using EVT_AccAddBias = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::plus,
          ElementCompute, ElementCompute, RoundStyle>,
      cutlass::epilogue::fusion::Sm90AccFetch,
      BiasLoad>;

  // SiLU(Acc + Bias)
  using EVT_Silu = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::epilogue::thread::SiLu,
          ElementCompute, ElementCompute, RoundStyle>,
      EVT_AccAddBias>;

  // SiLU(...) * Scale
  using EVT_Scaled = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies,
          ElementCompute, ElementCompute, RoundStyle>,
      EVT_Silu,
      ScaleLoad>;

  // ... + Aux
  using EVT_AddAux = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::plus,
          ElementCompute, ElementCompute, RoundStyle>,
      EVT_Scaled,
      AuxLoadViaC>;

  // tanh(...)
  using FusionCallbacks = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::epilogue::thread::Tanh,
          ElementD, ElementCompute, RoundStyle>,
      EVT_AddAux>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      FusionCallbacks
    >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

////////////////////////////////////////////////////////////////////////////////
// Runner.
//   A_bytes : torch.uint8 of length M*K   (one byte per E4M3 element, RowMajor)
//   B_bytes : torch.uint8 of length K*N   (one byte per E4M3 element, ColMajor)
//   bias_row, scale_col, Aux, D : torch.bfloat16
////////////////////////////////////////////////////////////////////////////////

void heavy_epi_impl(at::Tensor A_bytes, at::Tensor B_bytes,
                    at::Tensor bias_row, at::Tensor scale_col,
                    at::Tensor Aux, at::Tensor D,
                    int64_t M, int64_t N, int64_t K) {
  using Cfg          = GemmConfig;
  using Gemm         = Cfg::Gemm;
  using ElementA     = Cfg::ElementA;
  using ElementB     = Cfg::ElementB;
  using ElementC     = Cfg::ElementC;
  using ElementD     = Cfg::ElementD;
  using ElementBias  = Cfg::ElementBias;
  using ElementScale = Cfg::ElementScale;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  TORCH_CHECK(A_bytes.is_cuda() && B_bytes.is_cuda() && bias_row.is_cuda() &&
              scale_col.is_cuda() && Aux.is_cuda() && D.is_cuda(),
              "all inputs must be CUDA tensors");
  TORCH_CHECK(A_bytes.dtype() == at::kByte && B_bytes.dtype() == at::kByte,
              "A_bytes / B_bytes must be packed torch.uint8 (FP8 E4M3 = 1 byte/elt)");
  TORCH_CHECK(bias_row.scalar_type() == at::kBFloat16 &&
              scale_col.scalar_type() == at::kBFloat16 &&
              Aux.scalar_type()       == at::kBFloat16 &&
              D.scalar_type()         == at::kBFloat16,
              "bias_row, scale_col, Aux, D must all be bf16");
  TORCH_CHECK(A_bytes.is_contiguous() && B_bytes.is_contiguous() &&
              Aux.is_contiguous() && D.is_contiguous() &&
              bias_row.is_contiguous() && scale_col.is_contiguous(),
              "all tensors must be contiguous");

  TORCH_CHECK(A_bytes.numel() == M * K,
              "A packed byte count must equal M*K (1 byte/E4M3 element)");
  TORCH_CHECK(B_bytes.numel() == K * N,
              "B packed byte count must equal K*N (1 byte/E4M3 element)");
  TORCH_CHECK(bias_row.numel()  == N, "bias_row must have N elements");
  TORCH_CHECK(scale_col.numel() == M, "scale_col must have M elements");
  TORCH_CHECK(Aux.size(0) == M && Aux.size(1) == N, "Aux must be (M,N)");
  TORCH_CHECK(D.size(0)   == M && D.size(1)   == N, "D must be (M,N)");

  const c10::cuda::CUDAGuard guard(A_bytes.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(int(M), int(K), 1));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(int(N), int(K), 1));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(int(M), int(N), 1));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(int(M), int(N), 1));

  auto ptrA     = reinterpret_cast<ElementA const*>(A_bytes.data_ptr<uint8_t>());
  auto ptrB     = reinterpret_cast<ElementB const*>(B_bytes.data_ptr<uint8_t>());
  auto ptrAux_C = reinterpret_cast<ElementC const*>(Aux.data_ptr<at::BFloat16>());
  auto ptrD     = reinterpret_cast<ElementD*>(D.data_ptr<at::BFloat16>());
  auto ptrBias  = reinterpret_cast<ElementBias  const*>(bias_row.data_ptr<at::BFloat16>());
  auto ptrScale = reinterpret_cast<ElementScale const*>(scale_col.data_ptr<at::BFloat16>());

  // EVT nested arguments — depth-first through the tree:
  //   FusionCallbacks (Tanh)
  //     └── EVT_AddAux (plus)
  //           ├── EVT_Scaled (multiplies)
  //           │     ├── EVT_Silu
  //           │     │     └── EVT_AccAddBias (plus)
  //           │     │           ├── Sm90AccFetch         {}
  //           │     │           └── Sm90RowBroadcast     { ptrBias }
  //           │     │           op_args (plus)           {}
  //           │     │     op_args (silu)                 {}
  //           │     └── Sm90ColBroadcast                 { ptrScale }
  //           │     op_args (multiplies)                 {}
  //           └── Sm90SrcFetch                           {}   (uses ptr_C / Aux)
  //           op_args (plus)                             {}
  //     op_args (tanh)                                   {}
  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {int(M), int(N), int(K), 1},
    { ptrA, stride_A, ptrB, stride_B },
    { // epilogue args
      {   // FusionCallbacks (Tanh EVT)
        {   // EVT_AddAux (plus)
          {   // EVT_Scaled (multiplies)
            {   // EVT_Silu
              {   // EVT_AccAddBias (plus)
                {},                           // Sm90AccFetch leaf
                { ptrBias },                  // Sm90RowBroadcast leaf
                {}                            // plus op args
              },
              {}                              // SiLu op args
            },
            { ptrScale },                     // Sm90ColBroadcast leaf
            {}                                // multiplies op args
          },
          {},                                 // Sm90SrcFetch leaf (uses ptr_C)
          {}                                  // plus op args
        },
        {}                                    // Tanh op args
      },
      ptrAux_C, stride_C,                     // ptr_C / stride_C (Aux via C)
      ptrD,    stride_D                       // ptr_D / stride_D
    }
  };

  Gemm gemm_op;
  auto status = gemm_op.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "heavy_epi_120_90evt: can_implement failed: ",
              cutlassGetStatusString(status),
              " for (M,N,K)=(", M, ",", N, ",", K, ")");

  size_t workspace_size = Gemm::get_workspace_size(args);
  at::Tensor workspace = at::empty({static_cast<int64_t>(workspace_size)},
      at::TensorOptions().dtype(at::kByte).device(A_bytes.device()));

  status = gemm_op.initialize(args, workspace.data_ptr(), stream);
  if (status != cutlass::Status::kSuccess) {
    cudaError_t cuda_err = cudaGetLastError();
    TORCH_CHECK(false,
                "heavy_epi_120_90evt: initialize failed: ",
                cutlassGetStatusString(status),
                " | cudaGetLastError: ", cudaGetErrorString(cuda_err));
  }

  status = gemm_op.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "heavy_epi_120_90evt: run failed: ", cutlassGetStatusString(status));
}

} // namespace heavy_epi_120_90evt

////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUTLASS 3.x arch::Sm120 + Sm90EVT GEMM with heavy fused epilogue. "
            "A/B are FP8 E4M3 packed uint8; bias/scale/Aux/D are bf16. "
            "D = tanh(SiLU(A@B + bias_row) * scale_col + Aux).";
  m.def("heavy_epi_matmul_out",
        &heavy_epi_120_90evt::heavy_epi_impl,
        "D = tanh(SiLU(A@B + bias_row) * scale_col + Aux); "
        "A_bytes:(M*K,) uint8 (E4M3 RowMajor), "
        "B_bytes:(K*N,) uint8 (E4M3 ColMajor), "
        "bias_row:(N,) bf16, scale_col:(M,) bf16, Aux:(M,N) bf16, D:(M,N) bf16, "
        "M, N, K",
        pybind11::arg("A_bytes"),
        pybind11::arg("B_bytes"),
        pybind11::arg("bias_row"),
        pybind11::arg("scale_col"),
        pybind11::arg("Aux"),
        pybind11::arg("D"),
        pybind11::arg("M"),
        pybind11::arg("N"),
        pybind11::arg("K"));
}
