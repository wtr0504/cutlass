// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Minimum CUTLASS 3.x TMA + EVT demo — Sm120 (Blackwell-consumer, RTX 5090).
//
// Goal: a small, readable example that wires a TMA mainloop and a TMA
// epilogue together with an Epilogue Visitor Tree (EVT). Compared to the
// 5-node `heavy_epi_low_precision_torch_ext.cu`, this keeps the EVT
// shallow so the *structure* of the TMA + EVT integration is the focus.
//
// Hardware: Blackwell-consumer (sm_120, RTX 5090). The Sm120 dense MMA in
// CUTLASS today is the **F8F6F4 Tensor Core**, so A/B are FP8 E4M3 (passed
// as packed uint8). There is no Sm120 dense bf16 mainloop yet — that path
// only exists on Sm90 (Hopper) and `cudaFuncSetAttribute` from the Sm90
// kernel fails to initialize on Blackwell-consumer.
//
// Where TMA is used here:
//   * Mainloop  : Sm120 `KernelTmaWarpSpecialized*` schedule       → TMA loads A, B
//   * Epilogue  : `epilogue::TmaWarpSpecialized` (auto-picked)     → TMA loads C, TMA stores D
//   * EVT leaf  : `Sm90SrcFetch<ElementC>` reads the TMA-staged C tile
//
// Fused op:
//
//     D = ReLU( A @ B + alpha * C + bias_row )
//
//     A         : (M, K) FP8 E4M3 RowMajor   ── TMA-loaded by the mainloop  (packed uint8)
//     B         : (K, N) FP8 E4M3 ColMajor   ── TMA-loaded by the mainloop  (packed uint8, TN)
//     C         : (M, N) bf16    RowMajor    ── TMA-loaded by the epilogue
//     bias_row  : (N,)   bf16                ── per-column broadcast
//     alpha     : fp32 scalar
//     D         : (M, N) bf16    RowMajor    ── TMA-stored by the epilogue
//
// EVT tree (CUTLASS 3.x — `Sm90EVT`, reused on Sm120):
//
//     ReLU                                            (Sm90Compute<ReLu>)
//       └─ plus                                       (Sm90Compute<plus>)
//            ├─ plus                                  (Sm90Compute<plus>)
//            │     ├─ Sm90AccFetch                    (the WGMMA accumulator)
//            │     └─ multiplies(alpha, SrcFetch<C>)  (Sm90Compute<multiplies>
//            │                                         + Sm90ScalarBroadcast<α>
//            │                                         + Sm90SrcFetch<C>)
//            └─ Sm90RowBroadcast<bias_row>            (per-N broadcast)
//
// Accumulation is fp32, epilogue compute is fp32, output is bf16.
//
// Alignment (from `detail::get_input_alignment_bits` on FP8): K % 16 == 0.
// 128x128 TileShape adds M % 128 == 0 and N % 128 == 0.

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

namespace tma_evt_demo {

////////////////////////////////////////////////////////////////////////////////
// Compile-time GEMM configuration
////////////////////////////////////////////////////////////////////////////////

struct GemmConfig {
  // Sm120 dense MMA = F8F6F4. We pick FP8 E4M3, the simplest of the family.
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;

  using ElementC = cutlass::bfloat16_t;
  using ElementD = cutlass::bfloat16_t;
  using ElementBias    = cutlass::bfloat16_t;
  using ElementScalar  = float;
  using ElementAccumulator = float;
  using ElementCompute     = float;

  static constexpr int AlignmentA = 16;   // 128 bits / 8 bits per E4M3 element
  static constexpr int AlignmentB = 16;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;  // 8

  // Sm120 F8F6F4 builder requires TN: A RowMajor, B ColumnMajor (both K-major).
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutCTag = cutlass::layout::RowMajor;
  using LayoutDTag = cutlass::layout::RowMajor;

  using ArchTag       = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using TileShape    = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

  ////////////////////////////////////////////////////////////////////////////
  // EVT tree
  //
  //   ReLU
  //     └── plus
  //           ├── plus
  //           │     ├── Sm90AccFetch
  //           │     └── multiplies(ScalarBroadcast<α>, SrcFetch<C>)
  //           └── Sm90RowBroadcast<bias_row>
  ////////////////////////////////////////////////////////////////////////////

  using AccLeaf      = cutlass::epilogue::fusion::Sm90AccFetch;
  using SrcCLeaf     = cutlass::epilogue::fusion::Sm90SrcFetch<ElementC>;
  using AlphaLeaf    = cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementScalar>;
  using BiasRowLeaf  = cutlass::epilogue::fusion::Sm90RowBroadcast<
      /*Stages=*/0, TileShape, ElementBias, ElementCompute>;

  // alpha * C
  using EVT_AlphaC = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies,
          ElementCompute, ElementCompute, RoundStyle>,
      AlphaLeaf,
      SrcCLeaf>;

  // Acc + alpha * C
  using EVT_AccPlusAlphaC = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::plus,
          ElementCompute, ElementCompute, RoundStyle>,
      AccLeaf,
      EVT_AlphaC>;

  // (Acc + alpha * C) + bias_row
  using EVT_PlusBias = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::plus,
          ElementCompute, ElementCompute, RoundStyle>,
      EVT_AccPlusAlphaC,
      BiasRowLeaf>;

  // ReLU(...) — final compute; output type is ElementD (bf16)
  using FusionCallbacks = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::epilogue::thread::ReLu,
          ElementD, ElementCompute, RoundStyle>,
      EVT_PlusBias>;

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
// Runner
//
//   A_bytes : torch.uint8 of length M*K (one byte per E4M3 element), RowMajor
//   B_bytes : torch.uint8 of length K*N (one byte per E4M3 element), ColMajor
//   C, D, bias_row : torch.bfloat16
////////////////////////////////////////////////////////////////////////////////

void tma_evt_impl(at::Tensor A_bytes, at::Tensor B_bytes,
                  at::Tensor C, at::Tensor bias_row, double alpha,
                  at::Tensor D,
                  int64_t M, int64_t N, int64_t K) {
  using Cfg = GemmConfig;
  using Gemm = Cfg::Gemm;
  using ElementA    = Cfg::ElementA;
  using ElementB    = Cfg::ElementB;
  using ElementC    = Cfg::ElementC;
  using ElementD    = Cfg::ElementD;
  using ElementBias = Cfg::ElementBias;
  using ElementScalar = Cfg::ElementScalar;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  TORCH_CHECK(A_bytes.is_cuda() && B_bytes.is_cuda() && C.is_cuda()
                  && bias_row.is_cuda() && D.is_cuda(),
              "all inputs must be CUDA tensors");
  TORCH_CHECK(A_bytes.dtype() == at::kByte && B_bytes.dtype() == at::kByte,
              "A_bytes / B_bytes must be packed torch.uint8 (FP8 E4M3 = 1 byte/elt)");
  TORCH_CHECK(C.scalar_type() == at::kBFloat16
                  && bias_row.scalar_type() == at::kBFloat16
                  && D.scalar_type() == at::kBFloat16,
              "C, bias_row, D must all be bf16");
  TORCH_CHECK(A_bytes.is_contiguous() && B_bytes.is_contiguous()
                  && C.is_contiguous() && D.is_contiguous() && bias_row.is_contiguous(),
              "all tensors must be contiguous");

  TORCH_CHECK(A_bytes.numel() == M * K,
              "A packed byte count must equal M*K (1 byte/E4M3 element)");
  TORCH_CHECK(B_bytes.numel() == K * N,
              "B packed byte count must equal K*N (1 byte/E4M3 element)");
  TORCH_CHECK(C.size(0) == M && C.size(1) == N, "C must be (M,N)");
  TORCH_CHECK(D.size(0) == M && D.size(1) == N, "D must be (M,N)");
  TORCH_CHECK(bias_row.numel() == N, "bias_row must have N elements");

  const c10::cuda::CUDAGuard guard(A_bytes.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(int(M), int(K), 1));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(int(N), int(K), 1));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(int(M), int(N), 1));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(int(M), int(N), 1));

  auto ptrA    = reinterpret_cast<ElementA const*>(A_bytes.data_ptr<uint8_t>());
  auto ptrB    = reinterpret_cast<ElementB const*>(B_bytes.data_ptr<uint8_t>());
  auto ptrC    = reinterpret_cast<ElementC const*>(C.data_ptr<at::BFloat16>());
  auto ptrD    = reinterpret_cast<ElementD*>      (D.data_ptr<at::BFloat16>());
  auto ptrBias = reinterpret_cast<ElementBias const*>(bias_row.data_ptr<at::BFloat16>());

  ElementScalar const alpha_f = static_cast<ElementScalar>(alpha);

  // EVT arguments — depth-first through the tree:
  //   FusionCallbacks (ReLU)
  //     └── EVT_PlusBias (plus)
  //           ├── EVT_AccPlusAlphaC (plus)
  //           │     ├── Sm90AccFetch                 {}
  //           │     └── EVT_AlphaC (multiplies)
  //           │           ├── Sm90ScalarBroadcast    { {alpha} }
  //           │           └── Sm90SrcFetch<C>        {}
  //           │           op_args                    {}
  //           │     op_args                          {}
  //           └── Sm90RowBroadcast                   { ptrBias }
  //           op_args                                {}
  //     op_args (ReLU)                               {}
  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {int(M), int(N), int(K), 1},
    { ptrA, stride_A, ptrB, stride_B },
    {  // epilogue args
      {  // FusionCallbacks (ReLU EVT)
        {  // EVT_PlusBias (plus)
          {  // EVT_AccPlusAlphaC (plus)
            {},                              // Sm90AccFetch leaf
            {  // EVT_AlphaC (multiplies)
              { {alpha_f} },                 // Sm90ScalarBroadcast leaf
              {},                            // Sm90SrcFetch leaf (uses ptr_C)
              {}                             // multiplies op args
            },
            {}                               // outer plus op args
          },
          { ptrBias },                       // Sm90RowBroadcast leaf
          {}                                 // plus op args
        },
        {}                                   // ReLU op args
      },
      ptrC, stride_C,                        // ptr_C / stride_C (TMA-loaded)
      ptrD, stride_D                         // ptr_D / stride_D (TMA-stored)
    }
  };

  Gemm gemm_op;
  auto status = gemm_op.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "tma_evt_demo: can_implement failed: ",
              cutlassGetStatusString(status),
              " for (M,N,K)=(", M, ",", N, ",", K, ")");

  size_t workspace_size = Gemm::get_workspace_size(args);
  at::Tensor workspace = at::empty({static_cast<int64_t>(workspace_size)},
      at::TensorOptions().dtype(at::kByte).device(A_bytes.device()));

  status = gemm_op.initialize(args, workspace.data_ptr(), stream);
  if (status != cutlass::Status::kSuccess) {
    cudaError_t cuda_err = cudaGetLastError();
    TORCH_CHECK(false,
                "tma_evt_demo: initialize failed: ",
                cutlassGetStatusString(status),
                " | cudaGetLastError: ", cudaGetErrorString(cuda_err));
  }

  status = gemm_op.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "tma_evt_demo: run failed: ", cutlassGetStatusString(status));
}

} // namespace tma_evt_demo

////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Minimum CUTLASS 3.x TMA + EVT demo (Sm120, FP8 E4M3 inputs). "
            "D = ReLU(A @ B + alpha * C + bias_row).";
  m.def("tma_evt_matmul_out",
        &tma_evt_demo::tma_evt_impl,
        "D = ReLU(A @ B + alpha * C + bias_row); "
        "A_bytes:(M*K,) uint8 (E4M3 RowMajor), "
        "B_bytes:(K*N,) uint8 (E4M3 ColMajor), "
        "C:(M,N) bf16, bias_row:(N,) bf16, alpha:fp64 (cast to fp32), "
        "D:(M,N) bf16, M, N, K",
        pybind11::arg("A_bytes"),
        pybind11::arg("B_bytes"),
        pybind11::arg("C"),
        pybind11::arg("bias_row"),
        pybind11::arg("alpha"),
        pybind11::arg("D"),
        pybind11::arg("M"),
        pybind11::arg("N"),
        pybind11::arg("K"));
}
