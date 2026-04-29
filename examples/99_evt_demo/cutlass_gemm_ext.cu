// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

//
// Python-binding for a CUTLASS fp16 GEMM on Blackwell Geforce (sm_120).
//
// Exposes a single function
//
//   cutlass_gemm_ext.gemm_fp16(A: (M,K) fp16 CUDA, B: (N,K) fp16 CUDA) -> (M,N) fp16 CUDA
//
// which computes ``C = A @ B.T``, matching ``torch.mm(A, B.t())``. The kernel
// runs on the current CUDA stream, so torch.cuda.Event timers capture it.
//
// Implementation uses CUTLASS 2.x ``cutlass::gemm::device::Gemm`` with an
// Ampere (Sm80) TensorOp MMA (m16n8k16 fp16/fp16->fp32), which compiles and
// runs on sm_120 via backward compatibility. (The CUTLASS 3.x
// CollectiveBuilder for sm_120 currently only supports F8F6F4 MMA.)
//

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"

using ElementA   = cutlass::half_t;
using ElementB   = cutlass::half_t;
using ElementC   = cutlass::half_t;
using ElementAcc = float;

using LayoutA = cutlass::layout::RowMajor;     // A (M,K) row-major
using LayoutB = cutlass::layout::ColumnMajor;  // B (K,N) col-major == (N,K) row-major storage
using LayoutC = cutlass::layout::RowMajor;     // C (M,N) row-major

using MMAOp  = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

using TileShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape< 64,  64, 32>;
using MmaShape  = cutlass::gemm::GemmShape< 16,   8, 16>;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAcc,
    ElementAcc>;

using Swizzle   = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int NumStages = 4;

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAcc,
    MMAOp, SmArch,
    TileShape, WarpShape, MmaShape,
    EpilogueOp, Swizzle, NumStages>;

// Compute C = A @ B.T with A: (M,K) row-major fp16, B: (N,K) row-major fp16.
// Returns a freshly-allocated output tensor on the same device as the inputs.
at::Tensor cutlass_gemm_fp16(at::Tensor A, at::Tensor B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
  TORCH_CHECK(A.scalar_type() == at::kHalf && B.scalar_type() == at::kHalf,
              "A and B must be fp16 (torch.float16)");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
  TORCH_CHECK(A.size(1) == B.size(1),
              "A.size(1) (K) must equal B.size(1) (K), got ",
              A.sizes(), " vs ", B.sizes());
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
              "A and B must be contiguous (row-major)");

  int const M = static_cast<int>(A.size(0));
  int const K = static_cast<int>(A.size(1));
  int const N = static_cast<int>(B.size(0));

  auto opts = A.options();  // fp16 on same device
  at::Tensor C = at::empty({M, N}, opts);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();

  cutlass::gemm::GemmCoord problem{M, N, K};
  typename Gemm::Arguments args{
      problem,
      {reinterpret_cast<ElementA*>(A.data_ptr<at::Half>()), K},  // lda = K (RowMajor)
      {reinterpret_cast<ElementB*>(B.data_ptr<at::Half>()), K},  // ldb = K (ColumnMajor on (N,K) storage)
      {reinterpret_cast<ElementC*>(C.data_ptr<at::Half>()), N},
      {reinterpret_cast<ElementC*>(C.data_ptr<at::Half>()), N},
      {ElementAcc(1.0f), ElementAcc(0.0f)},
      /*split_k_slices=*/1};

  Gemm op;
  cutlass::Status st = op.initialize(args, /*workspace=*/nullptr, stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess,
              "CUTLASS init failed: ", cutlassGetStatusString(st),
              " for shape (M,N,K)=(", M, ",", N, ",", K, ")");
  st = op(stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess,
              "CUTLASS run failed: ", cutlassGetStatusString(st),
              " for shape (M,N,K)=(", M, ",", N, ",", K, ")");

  return C;
}

// Same as above but writes into a pre-allocated output tensor. Lets the
// benchmark reuse buffers and exactly match torch.mm(..., out=C) semantics.
void cutlass_gemm_fp16_out(at::Tensor A, at::Tensor B, at::Tensor C) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(),
              "A, B, C must all be CUDA tensors");
  TORCH_CHECK(A.scalar_type() == at::kHalf && B.scalar_type() == at::kHalf &&
                  C.scalar_type() == at::kHalf,
              "A, B, C must be fp16");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2,
              "A, B, C must be 2D");
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(),
              "A, B, C must be contiguous");
  TORCH_CHECK(A.size(1) == B.size(1), "K mismatch");
  TORCH_CHECK(C.size(0) == A.size(0) && C.size(1) == B.size(0),
              "C shape must be (M,N)");

  int const M = static_cast<int>(A.size(0));
  int const K = static_cast<int>(A.size(1));
  int const N = static_cast<int>(B.size(0));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();

  cutlass::gemm::GemmCoord problem{M, N, K};
  typename Gemm::Arguments args{
      problem,
      {reinterpret_cast<ElementA*>(A.data_ptr<at::Half>()), K},
      {reinterpret_cast<ElementB*>(B.data_ptr<at::Half>()), K},
      {reinterpret_cast<ElementC*>(C.data_ptr<at::Half>()), N},
      {reinterpret_cast<ElementC*>(C.data_ptr<at::Half>()), N},
      {ElementAcc(1.0f), ElementAcc(0.0f)},
      1};

  Gemm op;
  cutlass::Status st = op.initialize(args, nullptr, stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess,
              "CUTLASS init failed: ", cutlassGetStatusString(st));
  st = op(stream);
  TORCH_CHECK(st == cutlass::Status::kSuccess,
              "CUTLASS run failed: ", cutlassGetStatusString(st));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUTLASS fp16 Sm80 TensorOp GEMM (runs on sm_120)";
  m.def("gemm_fp16", &cutlass_gemm_fp16,
        "C = A @ B.T, A:(M,K) fp16, B:(N,K) fp16",
        pybind11::arg("A"), pybind11::arg("B"));
  m.def("gemm_fp16_out", &cutlass_gemm_fp16_out,
        "C = A @ B.T, in-place into preallocated C",
        pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"));
}
