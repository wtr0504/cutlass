// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Python-binding for a CUTLASS GEMM with the SAME "heavy" fused epilogue as
// `heavy_epi_torch_ext.cu`, but expressed with the **CUTLASS 3.x Sm90EVT**
// API (CollectiveBuilder + Sm90 fusion callbacks).
//
// IMPORTANT — hardware requirement: **NVIDIA Hopper (sm_90 / H100)**.
//
//   The Sm90 mainloop uses WGMMA (`wgmma.mma_async`, `wgmma.fence`, ...),
//   which CUTLASS gates on `__CUDA_ARCH__ == 900` *strictly* (see
//   `cutlass/arch/config.h: CUTLASS_ARCH_MMA_SM90_ENABLED`). Compiling for
//   sm_120 succeeds, but the WGMMA paths become device-side assertion stubs
//   ("Attempting to use wgmma.fence without CUTE_ARCH_MMA_SM90A_ENABLED"),
//   so launches on Blackwell-consumer (RTX 5090) or any non-Hopper card
//   abort. The benchmark driver detects this and only runs the Sm90 column
//   when get_device_capability() == (9, 0).
//
//   For the same fused epilogue on sm_120 the CUTLASS-3.x path lives in
//   `heavy_epi_low_precision_torch_ext.cu` (Sm120 F8F6F4 mainloop + same
//   Sm90EVT visitors). There is no CUTLASS 3.x dense bf16 mainloop for
//   sm_120 yet, which is why this file's Sm90 mainloop is the only sane
//   "Sm90EVT bf16" option.
//
// The fused computation is identical to the Sm80 version:
//
//     D = tanh( SiLU(A @ B + bias_row) * scale_col + Aux )
//
//     A         : (M, K) bf16 RowMajor
//     B         : (K, N) bf16 RowMajor      (same shape as the Sm80 entry)
//     bias_row  : (N,)   bf16     (broadcast along M)
//     scale_col : (M,)   bf16     (broadcast along N)
//     Aux       : (M, N) bf16 RowMajor   (bound to the GEMM "C" operand)
//     D         : (M, N) bf16 RowMajor
//
// Sm90 EVT tree (CUTLASS 3.x — `Sm90EVT`):
//
//     T0 = Acc + Bias             (Sm90AccFetch + Sm90RowBroadcast<Bias>)
//     T1 = SiLU(T0)               (Sm90Compute<SiLu>)
//     T2 = T1 * Scale             (Sm90ColBroadcast<Scale> * _)
//     T3 = T2 + SrcFetch(C=Aux)   (Sm90SrcFetch<C> serves as the Aux load)
//     D  = tanh(T3)               (Sm90Compute<Tanh>, final store)
//
// Accumulation is fp32, epilogue compute is fp32, output is bf16.
//
// Notes
//   * Layouts are TT (A RowMajor, B RowMajor), preserving Python-side input
//     parity with the Sm80 entry. KernelScheduleAuto picks an appropriate
//     warp-specialized schedule for bf16 + TT.
//   * ArchTag = Sm90 + ClusterShape<1,1,1>; the Python builder includes
//     `compute_90a/compute_90a` PTX so the runtime can JIT for sm_120+.
//   * "C" operand of the GEMM is repurposed as the Aux matrix (matching the
//     low-precision file). Sm90SrcFetch reads it through the epilogue's
//     existing C-load TMA path, which avoids manual Aux TMA plumbing.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/functional.h"
#include "cutlass/numeric_types.h"

#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cutlass/util/packed_stride.hpp"

////////////////////////////////////////////////////////////////////////////////

using namespace cute;

namespace heavy_epi_90 {

////////////////////////////////////////////////////////////////////////////////
// Compile-time GEMM configuration. All knobs (tile shape, EVT tree, schedule)
// are fixed; only the dtypes are bf16.
////////////////////////////////////////////////////////////////////////////////

struct GemmConfig {
  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 8
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 8

  // TT layout (matches Sm80 entry: B passed as (K, N) RowMajor).
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::RowMajor;

  using ElementC     = cutlass::bfloat16_t;     // bound to Aux via SrcFetch
  using ElementD     = cutlass::bfloat16_t;
  using ElementBias  = cutlass::bfloat16_t;
  using ElementScale = cutlass::bfloat16_t;
  using LayoutCTag   = cutlass::layout::RowMajor;
  using LayoutDTag   = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;  // 8

  using ElementAccumulator = float;
  using ElementCompute     = float;

  using ArchTag       = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using TileShape    = Shape<_128, _128, _64>;
  using ClusterShape = Shape<_1, _1, _1>;

  // Sm90 EVT fusion is rejected by EpilogueScheduleAuto (it only supports
  // plain LinearCombination); we must pin an explicit TmaWarpSpecialized
  // schedule. Pingpong is the right pair for ClusterShape <1,1,1>; the
  // cooperative variants would require Cluster_M >= 2.
  using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

  //
  // Heavy epilogue EVT (Sm90):
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

  // SiLU(...)
  using EVT_Silu = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<cutlass::epilogue::thread::SiLu,
          ElementCompute, ElementCompute, RoundStyle>,
      EVT_AccAddBias>;

  // ... * Scale
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
      EpilogueSchedule,
      FusionCallbacks
    >::CollectiveOp;

  // Pinned to a low stage count so the same cubin loads on both sm_90 and
  // sm_120: AutoCarveout would pick ~6 stages assuming Hopper's 228 KB
  // dynamic-smem budget; the resulting kernel then fails the host-side
  // cudaFuncSetAttribute call on sm_120 (~99 KB dyn-smem cap) with "no
  // kernel image is available for execution on the device". Note the kernel
  // still won't *run* on sm_120 because of the WGMMA arch gate (see header
  // comment) — this just keeps the loader happy.
  using StageCountType = cutlass::gemm::collective::StageCount<2>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator,
      TileShape, ClusterShape,
      StageCountType,
      KernelSchedule
    >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

////////////////////////////////////////////////////////////////////////////////
// Runner (single bf16 instantiation).
////////////////////////////////////////////////////////////////////////////////

void heavy_epi_impl(at::Tensor A, at::Tensor B,
                    at::Tensor bias_row, at::Tensor scale_col,
                    at::Tensor Aux, at::Tensor D) {
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

  TORCH_CHECK(A.is_cuda() && B.is_cuda() && bias_row.is_cuda() &&
              scale_col.is_cuda() && Aux.is_cuda() && D.is_cuda(),
              "all inputs must be CUDA tensors");
  TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16
                  && bias_row.scalar_type() == at::kBFloat16
                  && scale_col.scalar_type() == at::kBFloat16
                  && Aux.scalar_type() == at::kBFloat16
                  && D.scalar_type() == at::kBFloat16,
              "all inputs must be bf16 (torch.bfloat16)");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A, B must be 2D");
  TORCH_CHECK(A.size(1) == B.size(0), "K mismatch between A and B");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous() &&
              Aux.is_contiguous() && D.is_contiguous(),
              "A, B, Aux, D must be contiguous (row-major)");

  int const M = static_cast<int>(A.size(0));
  int const K = static_cast<int>(A.size(1));
  int const N = static_cast<int>(B.size(1));

  TORCH_CHECK(bias_row.numel()  == N, "bias_row must have N elements");
  TORCH_CHECK(scale_col.numel() == M, "scale_col must have M elements");
  TORCH_CHECK(Aux.size(0) == M && Aux.size(1) == N, "Aux must be (M,N)");
  TORCH_CHECK(D.size(0)   == M && D.size(1)   == N, "D must be (M,N)");

  const c10::cuda::CUDAGuard guard(A.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  // Packed strides — Shape is (M_or_N, K, L); the Stride type encodes layout.
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

  auto ptrA     = reinterpret_cast<ElementA const*>(A.data_ptr<at::BFloat16>());
  auto ptrB     = reinterpret_cast<ElementB const*>(B.data_ptr<at::BFloat16>());
  auto ptrAux_C = reinterpret_cast<ElementC const*>(Aux.data_ptr<at::BFloat16>());
  auto ptrD     = reinterpret_cast<ElementD*>      (D.data_ptr<at::BFloat16>());
  auto ptrBias  = reinterpret_cast<ElementBias  const*>(bias_row.data_ptr<at::BFloat16>());
  auto ptrScale = reinterpret_cast<ElementScale const*>(scale_col.data_ptr<at::BFloat16>());

  // Build EVT nested arguments. Brace layout mirrors the tree:
  //   outer: { FusionCallbacks_args, ptr_C, stride_C, ptr_D, stride_D }
  //   FusionCallbacks_args (Tanh): { AddAux_args, tanh_op_args }
  //     AddAux_args  (plus)  : { Scaled_args, SrcFetch_args, plus_op_args }
  //     Scaled_args  (mul)   : { Silu_args, ColBcast_args, mul_op_args }
  //     Silu_args    (silu)  : { AccAddBias_args, silu_op_args }
  //     AccAddBias_args(plus): { AccFetch_args, RowBcast_args, plus_op_args }
  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
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
              "heavy_epi_90: can_implement failed: ",
              cutlassGetStatusString(status),
              " for (M,N,K)=(", M, ",", N, ",", K, ")");

  size_t workspace_size = Gemm::get_workspace_size(args);
  at::Tensor workspace = at::empty({static_cast<int64_t>(workspace_size)},
      at::TensorOptions().dtype(at::kByte).device(A.device()));

  status = gemm_op.initialize(args, workspace.data_ptr(), stream);
  if (status != cutlass::Status::kSuccess) {
    cudaError_t cuda_err = cudaGetLastError();
    TORCH_CHECK(false,
                "heavy_epi_90: initialize failed: ",
                cutlassGetStatusString(status),
                " | cudaGetLastError: ", cudaGetErrorString(cuda_err));
  }

  status = gemm_op.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "heavy_epi_90: run failed: ", cutlassGetStatusString(status));
}

} // namespace heavy_epi_90

////////////////////////////////////////////////////////////////////////////////
// pybind11 module — same entry-point name as the Sm80 module (different
// PYBIND11_MODULE namespace), so the bench can call symmetrically.
////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUTLASS 3.x Sm90EVT GEMM with heavy fused epilogue (bf16). "
            "Built for compute_90a; runs natively on sm_90 and via PTX-JIT on sm_120+.";
  m.def("heavy_epi_matmul_out",
        &heavy_epi_90::heavy_epi_impl,
        "D = tanh(SiLU(A@B + bias_row) * scale_col + Aux); "
        "A:(M,K) bf16, B:(K,N) bf16, bias_row:(N,) bf16, "
        "scale_col:(M,) bf16, Aux:(M,N) bf16, D:(M,N) bf16",
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("bias_row"),
        pybind11::arg("scale_col"),
        pybind11::arg("Aux"),
        pybind11::arg("D"));
}
