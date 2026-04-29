// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Python-binding for a CUTLASS GEMM with a "heavy" fused epilogue on
// Blackwell Geforce (sm_120, RTX 5090).  Unlike `heavy_epi_torch_ext.cu`
// which uses bf16 inputs via the CUTLASS 2.x EVT API, this file uses the
// **CUTLASS 3.x CollectiveBuilder + Sm90EVT** path, which is required to
// reach the Blackwell narrow-precision (F8F6F4) Tensor Core MMA.
//
// A / B element types supported:
//   * FP8  E4M3  (cutlass::float_e4m3_t)
//   * FP8  E5M2  (cutlass::float_e5m2_t)
//   * FP6  E3M2  (cutlass::float_e3m2_t)
//   * FP6  E2M3  (cutlass::float_e2m3_t)
//   * FP4  E2M1  (cutlass::float_e2m1_t)
//
// The fused computation is the same as the bf16 reference:
//
//     D = tanh( SiLU(A @ B + bias_row) * scale_col + Aux )
//
//     A         : (M, K) low-precision, RowMajor (K-major)   [packed uint8]
//     B         : (K, N) low-precision, ColumnMajor (K-major) [packed uint8]
//     bias_row  : (N,)   bf16   (broadcast along M)
//     scale_col : (M,)   bf16   (broadcast along N)
//     Aux       : (M, N) bf16 RowMajor   (bound as the GEMM "C" operand)
//     D         : (M, N) bf16 RowMajor
//
// Because PyTorch has no native dtype for FP6/FP4 (and the API here stays
// uniform across all five dtypes), A and B are passed as **packed uint8
// buffers**: the i-th logical element lives in the bits
// `[i * sizeof_bits, (i+1) * sizeof_bits)` of the byte stream.  For FP8
// the packing degenerates to one element per byte and the caller can use
// `torch.float8_e4m3fn` / `torch.float8_e5m2` tensors viewed as `uint8`.
//
// Epilogue Visitor Tree (CUTLASS 3.x, `Sm90EVT`):
//
//     T0 = Acc + Bias             (AccFetch + RowBroadcast)
//     T1 = SiLU(T0)               (Compute<SiLu>)
//     T2 = T1 * Scale             (ColBroadcast * _)
//     T3 = T2 + SrcFetch(C=Aux)   (SrcFetch serves as Aux load)
//     D  = tanh(T3)               (Compute<Tanh>, final store)
//
// Accumulation is fp32, epilogue compute is fp32, output is bf16.
//
// Alignment requirements enforced by the builder
// (`detail::get_input_alignment_bits`):
//     FP8 :  A/B alignment = 16 elements (128 bits)
//     FP6 :  A/B alignment = 128 elements (96 bytes)
//     FP4 :  A/B alignment = 128 elements (64 bytes)
// The driver script should keep K divisible by the corresponding value.

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
#include "cutlass/float_subbyte.h"

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

namespace heavy_epi_lp {

////////////////////////////////////////////////////////////////////////////////
// Compile-time GEMM configuration parameterised by (ElementA, ElementB,
// AlignmentA, AlignmentB).  All other knobs (tile shape, epilogue dtypes,
// EVT tree, schedule) are fixed.
////////////////////////////////////////////////////////////////////////////////

template <class ElementA_, class ElementB_, int AlignmentA_, int AlignmentB_>
struct GemmConfig {
  using ElementA   = ElementA_;
  using ElementB   = ElementB_;
  static constexpr int AlignmentA = AlignmentA_;
  static constexpr int AlignmentB = AlignmentB_;

  // sm_120 F8F6F4 builder requires TN layout (A = RowMajor, B = ColumnMajor).
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutBTag = cutlass::layout::ColumnMajor;

  // Epilogue / fusion tensors.  "C" operand of the GEMM is repurposed as
  // the Aux matrix, which avoids having to wire up a bespoke Sm90AuxLoad
  // (its TMA layout atoms would otherwise need manual plumbing).
  using ElementC     = cutlass::bfloat16_t;
  using ElementD     = cutlass::bfloat16_t;
  using ElementBias  = cutlass::bfloat16_t;
  using ElementScale = cutlass::bfloat16_t;
  using LayoutCTag   = cutlass::layout::RowMajor;
  using LayoutDTag   = cutlass::layout::RowMajor;
  static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value; // 8
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value; // 8

  using ElementAccumulator = float;
  using ElementCompute     = float;

  using ArchTag       = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using TileShape    = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;           // required on sm_120

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

  //
  // Heavy epilogue EVT:
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
  using Plus  = cutlass::plus<ElementCompute>;
  using Mult  = cutlass::multiplies<ElementCompute>;

  // Leaf: row-broadcast bias, per-N
  using BiasLoad = cutlass::epilogue::fusion::Sm90RowBroadcast<
      /*Stages=*/0, TileShape, ElementBias, ElementCompute>;
  // Leaf: col-broadcast scale, per-M
  using ScaleLoad = cutlass::epilogue::fusion::Sm90ColBroadcast<
      /*Stages=*/0, TileShape, ElementScale, ElementCompute>;
  // Leaf: source-fetch C used as Aux (M,N) matrix
  using AuxLoadViaC = cutlass::epilogue::fusion::Sm90SrcFetch<ElementC>;

  // inner: Acc + Bias
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
// Generic runner.
//   A_bytes / B_bytes : torch.uint8, tightly-packed row-major / col-major
//                      narrow-precision data (bit-packed for FP6/FP4).
//   bias_row / scale_col / Aux / D : torch.bfloat16
////////////////////////////////////////////////////////////////////////////////

template <class Config>
void heavy_epi_impl(at::Tensor A_bytes, at::Tensor B_bytes,
                    at::Tensor bias_row, at::Tensor scale_col,
                    at::Tensor Aux, at::Tensor D,
                    int64_t M, int64_t N, int64_t K) {
  using Gemm         = typename Config::Gemm;
  using ElementA     = typename Config::ElementA;
  using ElementB     = typename Config::ElementB;
  using ElementC     = typename Config::ElementC;
  using ElementD     = typename Config::ElementD;
  using ElementBias  = typename Config::ElementBias;
  using ElementScale = typename Config::ElementScale;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  const c10::cuda::CUDAGuard guard(A_bytes.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(A_bytes.is_cuda() && B_bytes.is_cuda() && bias_row.is_cuda() &&
              scale_col.is_cuda() && Aux.is_cuda() && D.is_cuda(),
              "heavy_epi_low_precision: all tensors must be on CUDA");
  TORCH_CHECK(A_bytes.dtype() == at::kByte && B_bytes.dtype() == at::kByte,
              "A/B must be packed torch.uint8 buffers");
  TORCH_CHECK(bias_row.dtype() == at::kBFloat16 &&
              scale_col.dtype() == at::kBFloat16 &&
              Aux.dtype()       == at::kBFloat16 &&
              D.dtype()         == at::kBFloat16,
              "bias_row, scale_col, Aux, D must be torch.bfloat16");
  TORCH_CHECK(A_bytes.is_contiguous() && B_bytes.is_contiguous(),
              "A/B packed byte buffers must be contiguous");
  TORCH_CHECK(bias_row.is_contiguous() && scale_col.is_contiguous() &&
              Aux.is_contiguous() && D.is_contiguous(),
              "bias_row / scale_col / Aux / D must be contiguous");

  // Size checks: packed-byte length == logical_elements * sizeof_bits / 8.
  int64_t const bitsA = cutlass::sizeof_bits<ElementA>::value;
  int64_t const bitsB = cutlass::sizeof_bits<ElementB>::value;
  TORCH_CHECK(A_bytes.numel() * 8 == M * K * bitsA,
              "A packed byte count mismatch: got ", A_bytes.numel(),
              " bytes for M=", M, " K=", K, " bits=", bitsA);
  TORCH_CHECK(B_bytes.numel() * 8 == K * N * bitsB,
              "B packed byte count mismatch: got ", B_bytes.numel(),
              " bytes for K=", K, " N=", N, " bits=", bitsB);
  TORCH_CHECK(bias_row.numel()  == N, "bias_row must have N elements");
  TORCH_CHECK(scale_col.numel() == M, "scale_col must have M elements");
  TORCH_CHECK(Aux.numel() == M * N, "Aux must be M*N");
  TORCH_CHECK(D.numel()   == M * N, "D must be M*N");

  // Packed strides (RowMajor / ColumnMajor with k-major leading dim).
  auto stride_A = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(int(M), int(K), 1));
  auto stride_B = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(int(N), int(K), 1));
  auto stride_C = cutlass::make_cute_packed_stride(
      StrideC{}, cute::make_shape(int(M), int(N), 1));
  auto stride_D = cutlass::make_cute_packed_stride(
      StrideD{}, cute::make_shape(int(M), int(N), 1));

  auto ptrA     = reinterpret_cast<ElementA const*>(A_bytes.data_ptr<uint8_t>());
  auto ptrB     = reinterpret_cast<ElementB const*>(B_bytes.data_ptr<uint8_t>());
  auto ptrAux_C = reinterpret_cast<ElementC const*>(Aux.data_ptr<at::BFloat16>());
  auto ptrD     = reinterpret_cast<ElementD*>(D.data_ptr<at::BFloat16>());
  auto ptrBias  = reinterpret_cast<ElementBias  const*>(bias_row.data_ptr<at::BFloat16>());
  auto ptrScale = reinterpret_cast<ElementScale const*>(scale_col.data_ptr<at::BFloat16>());

  // Construct EVT nested arguments.  Brace layout mirrors the tree:
  //   outer: {AddAux_args, tanh_op_args}
  //   AddAux_args: {Scaled_args, SrcFetch_args, plus_op_args}
  //   Scaled_args: {Silu_args, ColBcast_args, mul_op_args}
  //   Silu_args  : {AccAddBias_args, silu_op_args}
  //   AccAddBias_args: {AccFetch_args, RowBcast_args, plus_op_args}
  typename Gemm::Arguments args{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {int(M), int(N), int(K), 1},
    { ptrA, stride_A, ptrB, stride_B },
    { // epilogue args
      {   // FusionCallbacks (tanh EVT) args
        {   // EVT_AddAux (plus) args
          {   // EVT_Scaled (multiplies) args
            {   // EVT_Silu args
              {   // EVT_AccAddBias (plus) args
                {},                           // Sm90AccFetch leaf (empty)
                { ptrBias },                  // Sm90RowBroadcast leaf
                {}                            // plus op args (empty)
              },
              {}                              // SiLu op args (no hyperparameters)
            },
            { ptrScale },                     // Sm90ColBroadcast leaf
            {}                                // multiplies op args (empty)
          },
          {},                                 // Sm90SrcFetch leaf (empty)
          {}                                  // plus op args (empty)
        },
        {}                                    // Tanh op args (empty)
      },
      ptrAux_C, stride_C,                     // epilogue ptr_C/stride_C (Aux)
      ptrD,    stride_D                       // epilogue ptr_D/stride_D
    }
  };

  Gemm gemm_op;
  auto status = gemm_op.can_implement(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "heavy_epi_low_precision: can_implement failed (status=",
              int(status),
              ") -- check M/N/K alignment and problem shape");

  size_t workspace_size = Gemm::get_workspace_size(args);
  auto workspace = at::empty({ static_cast<int64_t>(workspace_size) },
      at::TensorOptions().dtype(at::kByte).device(A_bytes.device()));

  status = gemm_op.initialize(args, workspace.data_ptr(), stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "heavy_epi_low_precision: initialize failed (status=",
              int(status), ")");

  status = gemm_op.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "heavy_epi_low_precision: run failed (status=",
              int(status), ")");
}

} // namespace heavy_epi_lp

////////////////////////////////////////////////////////////////////////////////
// Per-dtype entry points.
////////////////////////////////////////////////////////////////////////////////

#define DEFINE_HEAVY_EPI(FN, ELT_A, ELT_B, ALIGN_A, ALIGN_B)               \
  static void FN(at::Tensor A, at::Tensor B,                                \
                 at::Tensor bias_row, at::Tensor scale_col,                 \
                 at::Tensor Aux, at::Tensor D,                              \
                 int64_t M, int64_t N, int64_t K) {                         \
    using Cfg = heavy_epi_lp::GemmConfig<ELT_A, ELT_B, ALIGN_A, ALIGN_B>;   \
    heavy_epi_lp::heavy_epi_impl<Cfg>(A, B, bias_row, scale_col,            \
                                      Aux, D, M, N, K);                     \
  }

// Alignment in *elements* is picked so that AlignmentBits = SizeofBits*Align
// meets the F8F6F4 TMA requirement from
// `cutlass::detail::get_input_alignment_bits`:
//     FP8 (8b)  -> 128 bits  => 16 elements
//     FP6 (6b)  -> 768 bits  => 128 elements
//     FP4 (4b)  -> 512 bits  => 128 elements
DEFINE_HEAVY_EPI(heavy_epi_matmul_fp8_e4m3,
                 cutlass::float_e4m3_t, cutlass::float_e4m3_t, 16, 16)
DEFINE_HEAVY_EPI(heavy_epi_matmul_fp8_e5m2,
                 cutlass::float_e5m2_t, cutlass::float_e5m2_t, 16, 16)
DEFINE_HEAVY_EPI(heavy_epi_matmul_fp6_e3m2,
                 cutlass::float_e3m2_t, cutlass::float_e3m2_t, 128, 128)
DEFINE_HEAVY_EPI(heavy_epi_matmul_fp6_e2m3,
                 cutlass::float_e2m3_t, cutlass::float_e2m3_t, 128, 128)
DEFINE_HEAVY_EPI(heavy_epi_matmul_fp4_e2m1,
                 cutlass::float_e2m1_t, cutlass::float_e2m1_t, 128, 128)

#undef DEFINE_HEAVY_EPI

////////////////////////////////////////////////////////////////////////////////
// pybind11 module
////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUTLASS 3.x F8F6F4 GEMM with heavy fused EVT epilogue on sm_120. "
            "Epilogue: D = tanh(SiLU(A@B + bias_row) * scale_col + Aux). "
            "A/B are packed uint8 buffers; bias/scale/Aux/D are bf16.";

  // Signature convention for all entry points:
  //   A        : (M*K*bitsA/8,) torch.uint8, row-major bit-packed narrow-precision
  //   B        : (K*N*bitsB/8,) torch.uint8, col-major bit-packed narrow-precision
  //   bias_row : (N,)  torch.bfloat16
  //   scale_col: (M,)  torch.bfloat16
  //   Aux      : (M,N) torch.bfloat16 (bound to the GEMM C operand)
  //   D        : (M,N) torch.bfloat16
  //   M, N, K  : int64

  m.def("heavy_epi_matmul_fp8_e4m3", &heavy_epi_matmul_fp8_e4m3,
        "FP8 E4M3 GEMM with heavy fused epilogue (sm_120).");
  m.def("heavy_epi_matmul_fp8_e5m2", &heavy_epi_matmul_fp8_e5m2,
        "FP8 E5M2 GEMM with heavy fused epilogue (sm_120).");
  m.def("heavy_epi_matmul_fp6_e3m2", &heavy_epi_matmul_fp6_e3m2,
        "FP6 E3M2 GEMM with heavy fused epilogue (sm_120).");
  m.def("heavy_epi_matmul_fp6_e2m3", &heavy_epi_matmul_fp6_e2m3,
        "FP6 E2M3 GEMM with heavy fused epilogue (sm_120).");
  m.def("heavy_epi_matmul_fp4_e2m1", &heavy_epi_matmul_fp4_e2m1,
        "FP4 E2M1 GEMM with heavy fused epilogue (sm_120).");
}
