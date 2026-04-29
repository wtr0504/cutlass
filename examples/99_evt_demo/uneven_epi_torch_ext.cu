// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// "Uneven" per-chunk fused activation projection:
//
//   D[:,    :N1]      = silu(A @ W1.T)
//   D[:, N1:N1+N2]    = tanh(A @ W2.T)
//
//   A  : (M, K)        bf16 row-major
//   B  : (N1+N2, K)    bf16 row-major   (concat([W1, W2], dim=0))
//   D  : (M, N1+N2)    bf16 row-major
//
// Why we are NOT using DualGemm here
// ----------------------------------
// DualGemm (used by swiglu7_epi_one_stage) fuses two GEMMs that share A in
// smem. It requires the two GEMMs to have the SAME problem shape and the
// SAME threadblock tile — its public interface only takes a single
// `cutlass::gemm::GemmCoord problem` and a single ThreadblockShape, and the
// two B operands stride through smem in lockstep with A.
//
// Here N1 != N2, so the per-GEMM N differs and DualGemm does not apply.
//
// What we do instead
// ------------------
// Two independent CUTLASS GEMM kernel launches, each with its own
// activation epilogue (LinearCombinationSilu / LinearCombinationGeneric<Tanh>).
// Each writes directly into its slice of D (strided RowMajor with ldD = N).
//
// Trade-off vs DualGemm: A is read from HBM twice (once per launch)
// instead of once shared in smem. For compute-bound configurations the
// difference is small; for very memory-bound tails it is visible. We pay
// this cost because the alternative — a single big GEMM with a
// column-aware EVT epilogue that branches on the output column index —
// requires writing a custom Sm80EVT visitor and is the next step up in
// complexity.
//
// As in swiglu7_epi_one_stage, each chunk's W is viewed as a strided
// ColumnMajor matrix of shape (K, N_chunk) with ldB = K (its row stride
// in the original (N, K) row-major layout). That is the zero-copy way to
// express A @ W.T inside CUTLASS's (K, N) B-operand convention.
//
// Constraints
// -----------
//   K  % 8 == 0     (AlignmentA = AlignmentB = 8 for bf16)
//   N1 % 4 == 0     (AlignmentC = 4 — D's slice must be 8-byte aligned)
//   N2 % 4 == 0
//   N  % 4 == 0     (D's row stride must be 8-byte aligned)

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"

////////////////////////////////////////////////////////////////////////////////
// Data types and layouts
////////////////////////////////////////////////////////////////////////////////

using ElementA       = cutlass::bfloat16_t;
using ElementB       = cutlass::bfloat16_t;
using ElementC       = cutlass::bfloat16_t;
using ElementAcc     = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;   // (N_chunk, K) row-major == (K, N_chunk) col-major
using LayoutC = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // = 8
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // = 8
// AlignmentC = 4 (8 bytes) — D's slice base ptr is ptrD or ptrD + N1, and
// its row stride is N. Both must be 8-byte aligned. With AlignmentC = 4
// (4 bf16 elements = 8 bytes) we accept any N1, N divisible by 4.
constexpr int AlignmentC = 4;

////////////////////////////////////////////////////////////////////////////////
// Tile shapes — same as heavy_epi_torch_ext (Sm80 bf16 TensorOp)
////////////////////////////////////////////////////////////////////////////////

using ArchTag          = cutlass::arch::Sm80;
using OperatorClass    = cutlass::arch::OpClassTensorOp;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape        = cutlass::gemm::GemmShape< 64,  64, 32>;
using InstructionShape = cutlass::gemm::GemmShape< 16,   8, 16>;
constexpr int kStages = 4;
using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

////////////////////////////////////////////////////////////////////////////////
// Two epilogue ops — one per chunk, applied as part of the GEMM's epilogue.
//
// ScaleType::Nothing  =>  D = activation(Accum). No alpha/beta/source needed.
////////////////////////////////////////////////////////////////////////////////

constexpr auto kScaleType = cutlass::epilogue::thread::ScaleType::Nothing;

using EpilogueOpSilu = cutlass::epilogue::thread::LinearCombinationSilu<
    ElementC, AlignmentC, ElementAcc, ElementCompute, kScaleType>;

using EpilogueOpTanh = cutlass::epilogue::thread::LinearCombinationGeneric<
    cutlass::epilogue::thread::Tanh,
    ElementC, AlignmentC, ElementAcc, ElementCompute, kScaleType>;

////////////////////////////////////////////////////////////////////////////////
// Two device GEMM types — same mainloop, different epilogue activation
////////////////////////////////////////////////////////////////////////////////

using GemmSilu = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAcc,
    OperatorClass, ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOpSilu,
    Swizzle,
    kStages,
    AlignmentA, AlignmentB>;

using GemmTanh = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAcc,
    OperatorClass, ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOpTanh,
    Swizzle,
    kStages,
    AlignmentA, AlignmentB>;

////////////////////////////////////////////////////////////////////////////////
// Helper to launch one chunk
////////////////////////////////////////////////////////////////////////////////

template <typename GemmOp>
static void launch_chunk(ElementA const* ptrA,
                         ElementB const* ptrB_chunk,
                         ElementC*       ptrD_chunk,
                         int M, int N_chunk, int K, int64_t ldD,
                         at::Device dev,
                         cudaStream_t stream,
                         char const* name) {
    LayoutA layoutA(static_cast<int64_t>(K));
    LayoutB layoutB(static_cast<int64_t>(K));   // ldB = K (B chunk is (N_chunk, K) row-major == (K, N_chunk) col-major)
    LayoutC layoutD(ldD);                       // ldD = N (full output row stride)

    cutlass::TensorRef<ElementA const, LayoutA> ref_A(ptrA,        layoutA);
    cutlass::TensorRef<ElementB const, LayoutB> ref_B(ptrB_chunk,  layoutB);
    cutlass::TensorRef<ElementC const, LayoutC> ref_C(nullptr,     LayoutC(0));
    cutlass::TensorRef<ElementC,       LayoutC> ref_D(ptrD_chunk,  layoutD);

    typename GemmOp::EpilogueOutputOp::Params epi{
        ElementCompute(1.0f), ElementCompute(0.0f)};

    typename GemmOp::Arguments args(
        cutlass::gemm::GemmCoord{M, N_chunk, K},
        ref_A, ref_B, ref_C, ref_D,
        epi,
        /*split_k_slices=*/1);

    GemmOp op;

    cutlass::Status st = op.can_implement(args);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                name, " can_implement failed: ",
                cutlassGetStatusString(st),
                " for (M,N_chunk,K)=(", M, ",", N_chunk, ",", K, ")");

    size_t workspace_bytes = GemmOp::get_workspace_size(args);
    at::Tensor workspace;
    void* workspace_ptr = nullptr;
    if (workspace_bytes > 0) {
        workspace = at::empty({static_cast<int64_t>(workspace_bytes)},
                              at::TensorOptions().dtype(at::kByte).device(dev));
        workspace_ptr = workspace.data_ptr();
    }

    st = op.initialize(args, workspace_ptr, stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                name, " init failed: ", cutlassGetStatusString(st));
    st = op(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                name, " run failed: ", cutlassGetStatusString(st));
}

////////////////////////////////////////////////////////////////////////////////
// Python-facing launcher
////////////////////////////////////////////////////////////////////////////////

void uneven_epi_matmul_out(at::Tensor A,
                           at::Tensor B,
                           at::Tensor D,
                           int64_t N1) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && D.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == at::kBFloat16
                    && B.scalar_type() == at::kBFloat16
                    && D.scalar_type() == at::kBFloat16,
                "all inputs must be bf16 (torch.bfloat16)");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && D.dim() == 2, "A, B, D must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1),
                "K mismatch: A.size(1) must equal B.size(1)");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && D.is_contiguous(),
                "A, B, D must be contiguous (row-major)");

    int const M = static_cast<int>(A.size(0));
    int const K = static_cast<int>(A.size(1));
    int const N = static_cast<int>(B.size(0));

    TORCH_CHECK(N1 > 0 && N1 < N, "N1 must satisfy 0 < N1 < N (got N1=", N1,
                ", N=", N, ")");
    int const N1_i = static_cast<int>(N1);
    int const N2_i = N - N1_i;

    TORCH_CHECK(D.size(0) == M && D.size(1) == N,
                "D must be (M, N1+N2) = (", M, ",", N, "), got ", D.sizes());

    TORCH_CHECK((K  % AlignmentB) == 0,
                "K must be multiple of ", AlignmentB, " (got ", K, ")");
    TORCH_CHECK((N1_i % AlignmentC) == 0,
                "N1 must be multiple of ", AlignmentC, " (got ", N1_i, ")");
    TORCH_CHECK((N2_i % AlignmentC) == 0,
                "N2 must be multiple of ", AlignmentC, " (got ", N2_i, ")");

    auto ptrA = reinterpret_cast<ElementA*>(A.data_ptr<at::BFloat16>());
    auto ptrB = reinterpret_cast<ElementB*>(B.data_ptr<at::BFloat16>());
    auto ptrD = reinterpret_cast<ElementC*>(D.data_ptr<at::BFloat16>());

    int64_t const ldD = static_cast<int64_t>(N);   // D row stride

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();

    // Chunk 1: D[:, :N1] = silu(A @ W1.T), W1 = B[:N1, :]
    launch_chunk<GemmSilu>(
        ptrA, ptrB,                               // W1 starts at ptrB
        ptrD,                                     // D[:, 0:]
        M, N1_i, K, ldD, A.device(), stream, "GemmSilu");

    // Chunk 2: D[:, N1:N1+N2] = tanh(A @ W2.T), W2 = B[N1:, :]
    launch_chunk<GemmTanh>(
        ptrA, ptrB + static_cast<int64_t>(N1_i) * K,   // W2 starts N1*K elements in
        ptrD + N1_i,                              // D[:, N1:]
        M, N2_i, K, ldD, A.device(), stream, "GemmTanh");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUTLASS uneven per-chunk fused activation projection (bf16) on sm_120: "
              "D[:, :N1] = silu(A @ W1.T); D[:, N1:] = tanh(A @ W2.T)";
    m.def("uneven_epi_matmul_out",
          &uneven_epi_matmul_out,
          "Two-chunk fused projection with different activations and N1 != N2; "
          "A:(M,K) bf16, B:(N1+N2,K) bf16 (= concat([W1,W2], dim=0)), "
          "D:(M,N1+N2) bf16",
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("D"),
          pybind11::arg("N1"));
}
