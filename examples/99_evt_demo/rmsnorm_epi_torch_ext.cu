// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Fused matmul + RMSNorm via CUTLASS 2.x EVT (Sm80 multistage on sm_120).
//
//   D = rmsnorm(A @ B.T) * gamma
//
//   A     : (M, K)  bf16 row-major
//   B     : (N, K)  bf16 row-major   (torch.nn.Linear weight convention)
//   gamma : (N,)    bf16             (per-feature RMSNorm scale)
//   D     : (M, N)  bf16 row-major
//
// where, in pure PyTorch:
//
//   def rmsnorm(x, gamma, eps=1e-6):
//       rms_inv = (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
//       return x * rms_inv * gamma
//
// Two kernels:
//
//   Kernel 1 — GEMM with EVT side-effect row-reduction.
//     Computes Y = A @ B.T, in the same kernel:
//       (a) writes Y:(M,N) bf16 to HBM,
//       (b) accumulates per-row sum-of-squares into rms_buf:(M,) fp32 via
//           atomic_add (after smem reduction inside each threadblock).
//     EVT topology (two parallel paths from Accum joined by SelectFirst):
//
//       Accum ──── Square ── VisitorRowReduction<plus,atomic_add> ──┐
//          │                                                        ▼
//          └─────────────────────────────────────────────────► SelectFirst
//                                                                   │
//                                                              StoreY → Y
//
//     SelectFirst (binary compute) returns its first child unchanged so the
//     downstream StoreY sees the original (un-squared) accumulator. The
//     row-reduction child runs purely for its global-memory side effect.
//
//   Kernel 2 — Custom finalize+normalize:
//     rms_inv[m] = rsqrt(rms_buf[m] / N + eps)
//     D[m,n]    = bf16( float(Y[m,n]) * rms_inv[m] * float(gamma[n]) )
//
// vs. the obvious torch.compile path (cuBLAS matmul → Triton variance kernel
// → Triton normalize kernel), this saves: (1) the entire variance pass over
// Y (sum-of-squares is computed for free in the GEMM epilogue's register
// fragments) and (2) one kernel launch.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

using cute::_0;
using cute::_1;

////////////////////////////////////////////////////////////////////////////////
// Custom epilogue functors
//
// CUTLASS pattern: primary template is the scalar version, partial spec on
// cutlass::Array<T,N> is the SIMD-fragment version actually invoked by the
// epilogue.
////////////////////////////////////////////////////////////////////////////////

// square: x -> x*x
template <typename T>
struct SquareOp {
    CUTLASS_HOST_DEVICE
    T operator()(T const& x) const { return x * x; }
};

template <typename T, int N>
struct SquareOp<cutlass::Array<T, N>> {
    CUTLASS_HOST_DEVICE
    cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& x) const {
        SquareOp<T> op;
        cutlass::Array<T, N> out;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) out[i] = op(x[i]);
        return out;
    }
};

// select-first: (a, b) -> a    (binary; discards the second arg, used to
// drop the row-reduction passthrough so StoreY sees the original Accum)
template <typename T>
struct SelectFirstOp {
    CUTLASS_HOST_DEVICE
    T operator()(T const& a, T const& /*b*/) const { return a; }
};

template <typename T, int N>
struct SelectFirstOp<cutlass::Array<T, N>> {
    CUTLASS_HOST_DEVICE
    cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& a,
                                    cutlass::Array<T, N> const& /*b*/) const {
        return a;
    }
};

////////////////////////////////////////////////////////////////////////////////
// Data types and tile shapes
////////////////////////////////////////////////////////////////////////////////

using ElementA       = cutlass::bfloat16_t;
using ElementB       = cutlass::bfloat16_t;
using ElementC       = cutlass::bfloat16_t;   // Y / D dtype
using ElementAcc     = float;
using ElementCompute = float;
using ElementReduce  = float;                  // rms_buf dtype

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;  // (N,K) row-major == (K,N) col-major
using LayoutC = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // = 8
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 4;   // 8-byte stores; supports any row stride that's a multiple of 4 bf16

using ArchTag          = cutlass::arch::Sm80;
using OperatorClass    = cutlass::arch::OpClassTensorOp;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape        = cutlass::gemm::GemmShape< 64,  64, 32>;
using InstructionShape = cutlass::gemm::GemmShape< 16,   8, 16>;
constexpr int NumStages         = 4;
constexpr int EVTEpilogueStages = 1;

////////////////////////////////////////////////////////////////////////////////
// EVT
////////////////////////////////////////////////////////////////////////////////

using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape, WarpShape, ElementC, AlignmentC, EVTEpilogueStages>;

using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

// Square the accumulator (compute path)
using ComputeSquare = cutlass::epilogue::threadblock::VisitorCompute<
    SquareOp, ElementCompute, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using EVT_Square = cutlass::epilogue::threadblock::Sm80EVT<ComputeSquare, Accum>;

// Side-effect: collapse the N axis (sum over columns of the squared
// values), atomic-added into rms_buf[M]. Note CUTLASS's naming convention:
// `VisitorColReduction` reduces ALONG columns and produces a per-row
// vector — that's exactly the per-row sum-of-squares we want for RMSNorm.
// (`VisitorRowReduction` would do the opposite: per-column output.)
// Returns its input as passthrough; we never use the passthrough
// downstream — its purpose is the global-memory side effect.
using ColReduceSumSq = cutlass::epilogue::threadblock::VisitorColReduction<
    cutlass::plus,
    cutlass::atomic_add,
    OutputTileThreadMap,
    ElementReduce,
    ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<_1, _0, _0>>;     // default: M-stride=1 → per-row vector of length M

using EVT_RowReduce = cutlass::epilogue::threadblock::Sm80EVT<ColReduceSumSq, EVT_Square>;

// Binary "select first" — discards the row-reduce passthrough, returns
// the original accumulator value for storage.
using ComputeSelectFirst = cutlass::epilogue::threadblock::VisitorCompute<
    SelectFirstOp, ElementCompute, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using EVT_SelectFirst = cutlass::epilogue::threadblock::Sm80EVT<
    ComputeSelectFirst, Accum, EVT_RowReduce>;

// Store Y as bf16 (M,N) row-major
using StoreY = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementC,
    cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, _1, int64_t>>;

using EVT_Y = cutlass::epilogue::threadblock::Sm80EVT<StoreY, EVT_SelectFirst>;

////////////////////////////////////////////////////////////////////////////////
// Kernel 1 GEMM type
////////////////////////////////////////////////////////////////////////////////

using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
    ElementA, LayoutA, cutlass::ComplexTransform::kNone, AlignmentA,
    ElementB, LayoutB, cutlass::ComplexTransform::kNone, AlignmentB,
    ElementC, LayoutC, AlignmentC,
    ElementAcc,
    ElementCompute,
    OperatorClass,
    ArchTag,
    ThreadblockShape, WarpShape, InstructionShape,
    EVT_Y,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    cutlass::arch::OpMultiplyAdd,
    EVTEpilogueStages>::GemmKernel;

using DeviceGemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

////////////////////////////////////////////////////////////////////////////////
// Kernel 2: finalize + normalize
//
//   rms_inv[m] = rsqrt(rms_buf[m] * inv_N + eps)
//   D[m,n]     = bf16( float(Y[m,n]) * rms_inv[m] * float(gamma[n]) )
//
// Launch grid (n_tiles_per_row, M), block size BLOCK_N. Each block loads
// rms_buf[m] and broadcasts rms_inv via shared memory, then writes BLOCK_N
// columns of one row.
////////////////////////////////////////////////////////////////////////////////

constexpr int kFinalizeBlockN = 128;

__global__ void rmsnorm_finalize_kernel(
    __nv_bfloat16 const* __restrict__ Y,
    __nv_bfloat16 const* __restrict__ gamma,
    float         const* __restrict__ rms_buf,
    __nv_bfloat16*       __restrict__ D,
    int M, int N, float eps, float inv_N)
{
    int const m      = blockIdx.y;
    int const n_base = blockIdx.x * blockDim.x;
    int const n      = n_base + threadIdx.x;

    __shared__ float s_rms_inv;
    if (threadIdx.x == 0) {
        s_rms_inv = rsqrtf(rms_buf[m] * inv_N + eps);
    }
    __syncthreads();

    if (n < N) {
        int   const idx     = m * N + n;
        float const y       = __bfloat162float(Y[idx]);
        float const g       = __bfloat162float(gamma[n]);
        D[idx] = __float2bfloat16_rn(y * s_rms_inv * g);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Python-facing launcher
////////////////////////////////////////////////////////////////////////////////

// D = rmsnorm(A @ B.T) * gamma
//   A:(M,K) bf16, B:(N,K) bf16, gamma:(N,) bf16, D:(M,N) bf16
void rmsnorm_matmul_out(at::Tensor A,
                        at::Tensor B,
                        at::Tensor gamma,
                        at::Tensor D,
                        double eps) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && gamma.is_cuda() && D.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16
                    && gamma.scalar_type() == at::kBFloat16
                    && D.scalar_type() == at::kBFloat16,
                "all inputs must be bf16 (torch.bfloat16)");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && D.dim() == 2 && gamma.dim() == 1,
                "A, B, D must be 2D; gamma must be 1D");
    TORCH_CHECK(A.size(1) == B.size(1),
                "K mismatch: A.size(1) must equal B.size(1) (B is (N,K))");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous()
                    && gamma.is_contiguous() && D.is_contiguous(),
                "all inputs must be contiguous (row-major)");

    int const M = static_cast<int>(A.size(0));
    int const K = static_cast<int>(A.size(1));
    int const N = static_cast<int>(B.size(0));

    TORCH_CHECK(gamma.size(0) == N, "gamma must have shape (N,) = (", N, ",)");
    TORCH_CHECK(D.size(0) == M && D.size(1) == N,
                "D must be (M, N) = (", M, ",", N, "), got ", D.sizes());

    auto ptrA     = reinterpret_cast<ElementA*>(A.data_ptr<at::BFloat16>());
    auto ptrB     = reinterpret_cast<ElementB*>(B.data_ptr<at::BFloat16>());
    auto ptrGamma = reinterpret_cast<__nv_bfloat16*>(gamma.data_ptr<at::BFloat16>());
    auto ptrD     = reinterpret_cast<__nv_bfloat16*>(D.data_ptr<at::BFloat16>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();

    // ------------------------------------------------------------------
    // Allocate per-row sum-of-squares buffer (zeroed) and intermediate Y.
    // ------------------------------------------------------------------
    auto fp32_opts  = at::TensorOptions().dtype(at::kFloat).device(A.device());
    at::Tensor rms_buf = at::zeros({M}, fp32_opts);                  // (M,) fp32
    at::Tensor Y       = at::empty({M, N}, A.options());              // (M, N) bf16
    auto ptrRmsBuf = rms_buf.data_ptr<float>();
    auto ptrY      = reinterpret_cast<ElementC*>(Y.data_ptr<at::BFloat16>());

    int64_t const MN = static_cast<int64_t>(M) * static_cast<int64_t>(N);

    // ------------------------------------------------------------------
    // Kernel 1: EVT GEMM that writes Y and atomic-adds per-row sum(Y^2)
    // ------------------------------------------------------------------
    typename EVT_Y::Arguments callback_args{
        // EVT_SelectFirst: {child0=Accum, child1=EVT_RowReduce, op=ComputeSelectFirst}
        {
            // child0 = Accum (no args)
            {},
            // child1 = EVT_RowReduce: {child0=EVT_Square, op=RowReduceSumSq}
            {
                // child0 = EVT_Square: {child0=Accum, op=ComputeSquare}
                {
                    {},   // Accum
                    {}    // ComputeSquare (no args)
                },
                // RowReduceSumSq args: pointer, identity, stride
                {ptrRmsBuf, ElementReduce(0), {_1{}, _0{}, _0{}}}
            },
            {}    // ComputeSelectFirst (no args)
        },
        // StoreY args
        {reinterpret_cast<ElementC*>(ptrY), {int64_t(N), _1{}, MN}}
    };

    cutlass::gemm::GemmCoord problem{M, N, K};
    typename DeviceGemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        /*batch_count=*/1,
        callback_args,
        ptrA,
        ptrB,
        /*ptr_C=*/nullptr,
        /*ptr_D=*/nullptr,
        /*batch_stride_A=*/static_cast<int64_t>(M) * K,
        /*batch_stride_B=*/static_cast<int64_t>(N) * K,
        /*batch_stride_C=*/0,
        /*batch_stride_D=*/0,
        /*stride_a=*/static_cast<int64_t>(K),
        /*stride_b=*/static_cast<int64_t>(K),
        /*stride_c=*/0,
        /*stride_d=*/0);

    DeviceGemm op;
    cutlass::Status st = op.can_implement(args);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "rmsnorm GEMM can_implement failed: ",
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
                "rmsnorm GEMM init failed: ", cutlassGetStatusString(st));
    st = op(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "rmsnorm GEMM run failed: ", cutlassGetStatusString(st));

    // ------------------------------------------------------------------
    // Kernel 2: finalize (rsqrt) + normalize (Y * rms_inv * gamma) → D
    // ------------------------------------------------------------------
    float const inv_N = 1.0f / static_cast<float>(N);
    dim3 const block(kFinalizeBlockN);
    dim3 const grid((N + kFinalizeBlockN - 1) / kFinalizeBlockN, M);
    rmsnorm_finalize_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16 const*>(ptrY),
        ptrGamma,
        ptrRmsBuf,
        ptrD,
        M, N, static_cast<float>(eps), inv_N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUTLASS fused matmul + RMSNorm (bf16) on sm_120";
    m.def("rmsnorm_matmul_out",
          &rmsnorm_matmul_out,
          "D = rmsnorm(A @ B.T) * gamma; "
          "A:(M,K) bf16, B:(N,K) bf16, gamma:(N,) bf16, D:(M,N) bf16",
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("gamma"),
          pybind11::arg("D"),
          pybind11::arg("eps") = 1e-6);
}
