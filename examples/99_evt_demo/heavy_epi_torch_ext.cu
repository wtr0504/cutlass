// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Python-binding for a CUTLASS GEMM with a "heavy" fused epilogue on
// Blackwell Geforce (sm_120, RTX 5090). The epilogue is expressed with
// the CUTLASS 2.x Epilogue Visitor Tree (EVT) API.
//
// The fused computation is:
//
//   D = tanh( SiLU(A @ B + bias_row) * scale_col + Aux )
//
//   A         : (M, K) bf16 row-major
//   B         : (K, N) bf16 row-major
//   bias_row  : (N,)   bf16     (broadcast along M)
//   scale_col : (M,)   bf16     (broadcast along N)
//   Aux       : (M, N) bf16 row-major
//   D         : (M, N) bf16 row-major
//
// SiLU(x) = x * sigmoid(x). Accumulation is fp32, epilogue compute is fp32.
//
// EVT tree (bottom-up):
//   T0 = Accum + Bias          (VisitorRowBroadcast)
//   T1 = SiLU(T0)              (VisitorCompute<SiLu>, unary)
//   T2 = T1 * Scale            (VisitorColBroadcast then multiplies)
//   T3 = T2 + Aux              (VisitorAuxLoad then plus)
//   T4 = tanh(T3)              (VisitorCompute<Tanh>, unary)
//   D  = AuxStore(T4)
//
// The kernel is an Ampere (Sm80) TensorOp MMA that compiles and runs on
// sm_120 via backward compatibility.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/functional.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"

#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

using cute::_0;
using cute::_1;

////////////////////////////////////////////////////////////////////////////////
// Data types and layouts
////////////////////////////////////////////////////////////////////////////////

using ElementA       = cutlass::bfloat16_t;
using ElementB       = cutlass::bfloat16_t;
using ElementC       = cutlass::bfloat16_t;   // type of bias / scale / aux / D
using ElementAcc     = float;                  // tensor-core accumulator
using ElementCompute = float;                  // epilogue compute precision

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // = 8
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

////////////////////////////////////////////////////////////////////////////////
// Tile shapes for Sm80 bf16 TensorOp
////////////////////////////////////////////////////////////////////////////////

using ArchTag          = cutlass::arch::Sm80;
using OperatorClass    = cutlass::arch::OpClassTensorOp;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape        = cutlass::gemm::GemmShape< 64,  64, 32>;
using InstructionShape = cutlass::gemm::GemmShape< 16,   8, 16>;
constexpr int NumStages         = 4;
constexpr int EVTEpilogueStages = 1;

////////////////////////////////////////////////////////////////////////////////
// EVT (Epilogue Visitor Tree) definition
////////////////////////////////////////////////////////////////////////////////

using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape, WarpShape, ElementC, AlignmentC, EVTEpilogueStages>;

using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

// bias_row : (1, N) broadcast -> per-column bias
using BiasRow = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    OutputTileThreadMap, ElementC,
    cute::Stride<_0, _1, int32_t>>;

// scale_col : (M, 1) broadcast -> per-row scale
using ScaleCol = cutlass::epilogue::threadblock::VisitorColBroadcast<
    OutputTileThreadMap, ElementC,
    cute::Stride<_1, _0, int32_t>>;

// Aux : (M, N) row-major
using Aux = cutlass::epilogue::threadblock::VisitorAuxLoad<
    OutputTileThreadMap, ElementC,
    cute::Stride<int64_t, _1, int64_t>>;

using ComputeAdd = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::plus, ElementCompute, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using ComputeMul = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, ElementCompute, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using ComputeSiLu = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::epilogue::thread::SiLu, ElementCompute, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using ComputeTanh = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::epilogue::thread::Tanh, ElementCompute, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

// T0 = Accum + BiasRow
using EVT_AddBias = cutlass::epilogue::threadblock::Sm80EVT<
    ComputeAdd, Accum, BiasRow>;

// T1 = SiLU(T0)
using EVT_SiLu = cutlass::epilogue::threadblock::Sm80EVT<
    ComputeSiLu, EVT_AddBias>;

// T2 = T1 * ScaleCol
using EVT_Scale = cutlass::epilogue::threadblock::Sm80EVT<
    ComputeMul, EVT_SiLu, ScaleCol>;

// T3 = T2 + Aux
using EVT_AddAux = cutlass::epilogue::threadblock::Sm80EVT<
    ComputeAdd, EVT_Scale, Aux>;

// T4 = tanh(T3)
using EVT_Tanh = cutlass::epilogue::threadblock::Sm80EVT<
    ComputeTanh, EVT_AddAux>;

// D : (M, N) row-major store
using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementC,
    cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, _1, int64_t>>;

using EVT_D = cutlass::epilogue::threadblock::Sm80EVT<
    StoreD, EVT_Tanh>;

////////////////////////////////////////////////////////////////////////////////
// Kernel / device GEMM type
////////////////////////////////////////////////////////////////////////////////

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
    ElementC, LayoutC, AlignmentC,
    ElementAcc,
    ElementCompute,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EVT_D,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    cutlass::arch::OpMultiplyAdd,
    EVTEpilogueStages>::GemmKernel;

using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

////////////////////////////////////////////////////////////////////////////////
// Python-facing launcher
////////////////////////////////////////////////////////////////////////////////

//   D = tanh( SiLU(A @ B + bias_row) * scale_col + Aux )
void heavy_epi_matmul_out(at::Tensor A,
                          at::Tensor B,
                          at::Tensor bias_row,
                          at::Tensor scale_col,
                          at::Tensor Aux,
                          at::Tensor D) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && bias_row.is_cuda()
                  && scale_col.is_cuda() && Aux.is_cuda() && D.is_cuda(),
              "all inputs must be CUDA tensors");
  TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16
                  && bias_row.scalar_type() == at::kBFloat16
                  && scale_col.scalar_type() == at::kBFloat16
                  && Aux.scalar_type() == at::kBFloat16
                  && D.scalar_type() == at::kBFloat16,
              "all inputs must be bf16 (torch.bfloat16)");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A, B must be 2D");
  TORCH_CHECK(A.size(1) == B.size(0), "K mismatch between A and B");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous()
                  && Aux.is_contiguous() && D.is_contiguous(),
              "A, B, Aux, D must be contiguous (row-major)");

  int const M = static_cast<int>(A.size(0));
  int const K = static_cast<int>(A.size(1));
  int const N = static_cast<int>(B.size(1));

  TORCH_CHECK(bias_row.numel()  == N, "bias_row must have N elements");
  TORCH_CHECK(scale_col.numel() == M, "scale_col must have M elements");
  TORCH_CHECK(Aux.size(0) == M && Aux.size(1) == N, "Aux must be (M,N)");
  TORCH_CHECK(D.size(0)   == M && D.size(1)   == N, "D must be (M,N)");

  auto ptrA    = reinterpret_cast<ElementA*>(A.data_ptr<at::BFloat16>());
  auto ptrB    = reinterpret_cast<ElementB*>(B.data_ptr<at::BFloat16>());
  auto ptrBias = reinterpret_cast<ElementC*>(bias_row.data_ptr<at::BFloat16>());
  auto ptrScl  = reinterpret_cast<ElementC*>(scale_col.data_ptr<at::BFloat16>());
  auto ptrAux  = reinterpret_cast<ElementC*>(Aux.data_ptr<at::BFloat16>());
  auto ptrD    = reinterpret_cast<ElementC*>(D.data_ptr<at::BFloat16>());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();

  // Build the EVT arguments (depth-first through the tree).
  int64_t const MN = static_cast<int64_t>(M) * static_cast<int64_t>(N);
  typename EVT_D::Arguments callback_args{
      { // EVT_Tanh
        { // EVT_AddAux
          { // EVT_Scale
            { // EVT_SiLu
              { // EVT_AddBias
                {},                                                     // Accum
                {ptrBias, ElementC(0), {_0{}, _1{}, int32_t(N)}},       // BiasRow
                {}                                                      // ComputeAdd
              },
              {}                                                        // ComputeSiLu
            },
            {ptrScl, ElementC(0), {_1{}, _0{}, int32_t(M)}},            // ScaleCol
            {}                                                          // ComputeMul
          },
          {ptrAux, ElementC(0), {int64_t(N), _1{}, MN}},                // Aux
          {}                                                            // ComputeAdd
        },
        {}                                                              // ComputeTanh
      },
      {ptrD, {int64_t(N), _1{}, MN}},                                   // StoreD
  };

  cutlass::gemm::GemmCoord problem{M, N, K};
  typename DeviceGemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      /*batch_count=*/1,
      callback_args,
      ptrA,                                        // ptr_A
      ptrB,                                        // ptr_B
      /*ptr_C=*/nullptr,
      /*ptr_D=*/nullptr,
      /*batch_stride_A=*/static_cast<int64_t>(M) * K,
      /*batch_stride_B=*/static_cast<int64_t>(K) * N,
      /*batch_stride_C=*/0,
      /*batch_stride_D=*/0,
      /*stride_a=*/static_cast<int64_t>(K),        // ldA (RowMajor M x K)
      /*stride_b=*/static_cast<int64_t>(N),        // ldB (RowMajor K x N)
      /*stride_c=*/0,
      /*stride_d=*/0);

  DeviceGemm op;

  cutlass::Status st = op.can_implement(args);
  TORCH_CHECK(st == cutlass::Status::kSuccess,
              "CUTLASS can_implement failed: ",
              cutlassGetStatusString(st),
              " for (M,N,K)=(", M, ",", N, ",", K, ")");

  size_t workspace_bytes = DeviceGemm::get_workspace_size(args);
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_bytes > 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_bytes)},
                          at::TensorOptions().dtype(at::kByte).device(A.device()));
    workspace_ptr = workspace.data_ptr();
  }

  st = op.initialize(args, workspace_ptr, stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess,
              "CUTLASS init failed: ", cutlassGetStatusString(st));
  st = op(stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess,
              "CUTLASS run failed: ", cutlassGetStatusString(st));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUTLASS EVT GEMM with heavy fused epilogue (bf16) on sm_120";
  m.def("heavy_epi_matmul_out",
        &heavy_epi_matmul_out,
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
