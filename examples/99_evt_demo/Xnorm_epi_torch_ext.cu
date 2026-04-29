// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Generic fused matmul + XXXnorm via CUTLASS 2.x EVT (Sm80 multistage on sm_120).
//
//   D = norm(A @ B.T) * gamma [+ beta]
//
//   A     : (M, K)  bf16 row-major
//   B     : (N, K)  bf16 row-major   (torch.nn.Linear weight convention)
//   gamma : (N,)    bf16             (per-feature affine scale, optional)
//   beta  : (N,)    bf16             (per-feature affine bias, optional, LayerNorm only)
//   D     : (M, N)  bf16 row-major
//
// Supported norm types (selected at launch time, dispatched in finalize kernel):
//
//   RMSNorm   : D = Y * rsqrt(mean(Y^2) + eps) * gamma
//   LayerNorm : D = (Y - mean(Y)) * rsqrt(var(Y) + eps) * gamma + beta
//   L2Norm    : D = Y * rsqrt(sum(Y^2) + eps) * gamma     (no /N)
//
// Two kernels:
//
//   Kernel 1 — GEMM with EVT side-effect dual row-reductions.
//     Computes Y = A @ B.T in registers, in the same kernel:
//       (a) writes Y:(M,N) bf16 to HBM,
//       (b) accumulates per-row sum-of-squares  into sumsq_buf:(M,) fp32,
//       (c) accumulates per-row sum            into sum_buf  :(M,) fp32,
//     both via atomic_add (after smem reduction inside each threadblock).
//
//     EVT topology — two side-effect ColReductions chained off Accum:
//
//        Accum ── Square ── ColReduce(sumsq) ──┐
//          │                                   ▼
//          └──────────────── SelectFirst(Accum, _) ── ColReduce(sum) ── StoreY → Y
//
//     Both ColReductions return their input as passthrough; SelectFirst drops
//     the squared value so the sum reduction (and StoreY) see the original
//     accumulator. The two reductions are nearly free — they reuse register
//     fragments already produced by the GEMM and add one extra atomic per tile.
//
//   Kernel 2 — Custom finalize+normalize, templated on NormType:
//     For RMSNorm   : inv_std = rsqrt(sumsq/N + eps);                D = Y * inv_std * gamma
//     For LayerNorm : mean    = sum/N;
//                     var     = sumsq/N - mean^2;
//                     inv_std = rsqrt(var + eps);                    D = (Y-mean)*inv_std*gamma + beta
//     For L2Norm    : inv_std = rsqrt(sumsq + eps);                  D = Y * inv_std * gamma
//
// vs. the obvious torch.compile path (cuBLAS matmul → Triton variance kernel
// → Triton normalize kernel), this saves: (1) the entire variance pass over Y
// (mean / sum-of-squares are computed for free in the GEMM epilogue's register
// fragments) and (2) one full kernel launch.

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
// Custom epilogue functors (scalar + Array<T,N> SIMD-fragment specializations).
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
// drop a side-branch passthrough so downstream sees the original Accum)
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
// Norm type tag (kept in sync with Python side).
////////////////////////////////////////////////////////////////////////////////

enum class NormType : int {
    RMSNorm   = 0,
    LayerNorm = 1,
    L2Norm    = 2,
};

////////////////////////////////////////////////////////////////////////////////
// Data types and tile shapes
////////////////////////////////////////////////////////////////////////////////

using ElementA       = cutlass::bfloat16_t;
using ElementB       = cutlass::bfloat16_t;
using ElementC       = cutlass::bfloat16_t;   // Y / D dtype
using ElementAcc     = float;
using ElementCompute = float;
using ElementReduce  = float;                  // sum / sumsq buf dtype

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;  // (N,K) row-major == (K,N) col-major
using LayoutC = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // = 8
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentC = 4;

using ArchTag          = cutlass::arch::Sm80;
using OperatorClass    = cutlass::arch::OpClassTensorOp;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape        = cutlass::gemm::GemmShape< 64,  64, 32>;
using InstructionShape = cutlass::gemm::GemmShape< 16,   8, 16>;
constexpr int NumStages         = 4;
constexpr int EVTEpilogueStages = 1;

////////////////////////////////////////////////////////////////////////////////
// EVT — two side-effect column reductions (one per row stat) chained off Accum.
//
//   Accum ── Square ── ColReduce(sumsq) ── SelectFirst(Accum,_) ── ColReduce(sum) ── StoreY
//
// The Square branch stays "live" only inside the sumsq reduction; SelectFirst
// drops it. The sum reduction takes the (un-squared) Accum directly and also
// passes it through to StoreY.
////////////////////////////////////////////////////////////////////////////////

using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape, WarpShape, ElementC, AlignmentC, EVTEpilogueStages>;

using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

// --- sumsq side branch -----------------------------------------------------
using ComputeSquare = cutlass::epilogue::threadblock::VisitorCompute<
    SquareOp, ElementCompute, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using EVT_Square = cutlass::epilogue::threadblock::Sm80EVT<ComputeSquare, Accum>;

// VisitorColReduction reduces ALONG columns -> per-row vector of length M.
using ColReduceSumSq = cutlass::epilogue::threadblock::VisitorColReduction<
    cutlass::plus,
    cutlass::atomic_add,
    OutputTileThreadMap,
    ElementReduce,
    ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<_1, _0, _0>>;

using EVT_RowReduceSq = cutlass::epilogue::threadblock::Sm80EVT<ColReduceSumSq, EVT_Square>;

// --- merge: drop the squared passthrough, recover Accum --------------------
using ComputeSelectFirst = cutlass::epilogue::threadblock::VisitorCompute<
    SelectFirstOp, ElementCompute, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using EVT_DropSq = cutlass::epilogue::threadblock::Sm80EVT<
    ComputeSelectFirst, Accum, EVT_RowReduceSq>;

// --- sum side branch (chained on top, sees raw Accum via DropSq) -----------
using ColReduceSum = cutlass::epilogue::threadblock::VisitorColReduction<
    cutlass::plus,
    cutlass::atomic_add,
    OutputTileThreadMap,
    ElementReduce,
    ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<_1, _0, _0>>;

using EVT_RowReduceSum = cutlass::epilogue::threadblock::Sm80EVT<ColReduceSum, EVT_DropSq>;

// --- store Y ---------------------------------------------------------------
using StoreY = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementC,
    cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, _1, int64_t>>;

using EVT_Y = cutlass::epilogue::threadblock::Sm80EVT<StoreY, EVT_RowReduceSum>;

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
// Kernel 2: finalize + normalize, templated on NormType.
//
// Launch grid (n_tiles_per_row, M), block size BLOCK_N. Each block loads
// sum_buf[m] and sumsq_buf[m] once, broadcasts (mean, inv_std) via shared
// memory, then writes BLOCK_N columns of one row.
////////////////////////////////////////////////////////////////////////////////

constexpr int kFinalizeBlockN = 128;

template <NormType NORM>
__global__ void xnorm_finalize_kernel(
    __nv_bfloat16 const* __restrict__ Y,
    __nv_bfloat16 const* __restrict__ gamma,    // may be nullptr
    __nv_bfloat16 const* __restrict__ beta,     // may be nullptr (LayerNorm only)
    float         const* __restrict__ sum_buf,
    float         const* __restrict__ sumsq_buf,
    __nv_bfloat16*       __restrict__ D,
    int M, int N, float eps, float inv_N)
{
    int const m      = blockIdx.y;
    int const n_base = blockIdx.x * blockDim.x;
    int const n      = n_base + threadIdx.x;

    __shared__ float s_inv_std;
    __shared__ float s_mean;
    if (threadIdx.x == 0) {
        float const sum   = sum_buf  [m];
        float const sumsq = sumsq_buf[m];
        if constexpr (NORM == NormType::RMSNorm) {
            s_mean    = 0.0f;
            s_inv_std = rsqrtf(sumsq * inv_N + eps);
        } else if constexpr (NORM == NormType::LayerNorm) {
            float const mean = sum * inv_N;
            float const var  = sumsq * inv_N - mean * mean;
            s_mean    = mean;
            s_inv_std = rsqrtf(var + eps);
        } else { // L2Norm
            s_mean    = 0.0f;
            s_inv_std = rsqrtf(sumsq + eps);
        }
    }
    __syncthreads();

    if (n < N) {
        int   const idx = m * N + n;
        float const y   = __bfloat162float(Y[idx]);
        float const g   = (gamma != nullptr) ? __bfloat162float(gamma[n]) : 1.0f;
        float       out;
        if constexpr (NORM == NormType::LayerNorm) {
            float const b = (beta != nullptr) ? __bfloat162float(beta[n]) : 0.0f;
            out = (y - s_mean) * s_inv_std * g + b;
        } else {
            out = y * s_inv_std * g;
        }
        D[idx] = __float2bfloat16_rn(out);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Python-facing launcher
////////////////////////////////////////////////////////////////////////////////

// D = norm(A @ B.T) * gamma [+ beta]
//
//   A:(M,K) bf16, B:(N,K) bf16, gamma:(N,) bf16 or None, beta:(N,) bf16 or None,
//   D:(M,N) bf16, norm_type ∈ {"rmsnorm", "layernorm", "l2norm"}.
void xnorm_matmul_out(at::Tensor              A,
                      at::Tensor              B,
                      c10::optional<at::Tensor> gamma_opt,
                      c10::optional<at::Tensor> beta_opt,
                      at::Tensor              D,
                      std::string             norm_type,
                      double                  eps) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && D.is_cuda(),
                "A, B, D must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16
                    && D.scalar_type() == at::kBFloat16,
                "A, B, D must be bf16 (torch.bfloat16)");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && D.dim() == 2,
                "A, B, D must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1),
                "K mismatch: A.size(1) must equal B.size(1) (B is (N,K))");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && D.is_contiguous(),
                "A, B, D must be contiguous (row-major)");

    int const M = static_cast<int>(A.size(0));
    int const K = static_cast<int>(A.size(1));
    int const N = static_cast<int>(B.size(0));
    TORCH_CHECK(D.size(0) == M && D.size(1) == N,
                "D must be (M, N) = (", M, ",", N, "), got ", D.sizes());

    // Decode norm-type tag (lowercased).
    NormType norm_kind;
    if      (norm_type == "rmsnorm")   norm_kind = NormType::RMSNorm;
    else if (norm_type == "layernorm") norm_kind = NormType::LayerNorm;
    else if (norm_type == "l2norm")    norm_kind = NormType::L2Norm;
    else TORCH_CHECK(false, "norm_type must be one of {rmsnorm, layernorm, l2norm}, "
                            "got '", norm_type, "'");

    // gamma / beta shape & dtype checks.
    __nv_bfloat16* ptrGamma = nullptr;
    if (gamma_opt.has_value()) {
        auto const& g = gamma_opt.value();
        TORCH_CHECK(g.is_cuda() && g.scalar_type() == at::kBFloat16,
                    "gamma must be bf16 CUDA");
        TORCH_CHECK(g.dim() == 1 && g.size(0) == N && g.is_contiguous(),
                    "gamma must be contiguous (N,)");
        ptrGamma = reinterpret_cast<__nv_bfloat16*>(g.data_ptr<at::BFloat16>());
    }
    __nv_bfloat16* ptrBeta = nullptr;
    if (beta_opt.has_value()) {
        TORCH_CHECK(norm_kind == NormType::LayerNorm,
                    "beta is only meaningful for layernorm");
        auto const& b = beta_opt.value();
        TORCH_CHECK(b.is_cuda() && b.scalar_type() == at::kBFloat16,
                    "beta must be bf16 CUDA");
        TORCH_CHECK(b.dim() == 1 && b.size(0) == N && b.is_contiguous(),
                    "beta must be contiguous (N,)");
        ptrBeta = reinterpret_cast<__nv_bfloat16*>(b.data_ptr<at::BFloat16>());
    }

    auto ptrA = reinterpret_cast<ElementA*>(A.data_ptr<at::BFloat16>());
    auto ptrB = reinterpret_cast<ElementB*>(B.data_ptr<at::BFloat16>());
    auto ptrD = reinterpret_cast<__nv_bfloat16*>(D.data_ptr<at::BFloat16>());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();

    // ------------------------------------------------------------------
    // Allocate per-row stat buffers (zeroed) and intermediate Y.
    // ------------------------------------------------------------------
    auto fp32_opts = at::TensorOptions().dtype(at::kFloat).device(A.device());
    at::Tensor sum_buf   = at::zeros({M}, fp32_opts);    // (M,) fp32
    at::Tensor sumsq_buf = at::zeros({M}, fp32_opts);    // (M,) fp32
    at::Tensor Y         = at::empty({M, N}, A.options()); // (M,N) bf16
    auto ptrSum   = sum_buf  .data_ptr<float>();
    auto ptrSumSq = sumsq_buf.data_ptr<float>();
    auto ptrY     = reinterpret_cast<ElementC*>(Y.data_ptr<at::BFloat16>());

    int64_t const MN = static_cast<int64_t>(M) * static_cast<int64_t>(N);

    // ------------------------------------------------------------------
    // Kernel 1: EVT GEMM that writes Y, atomic-adds per-row sum and sumsq.
    //
    // Visitor tree (top-down):
    //   StoreY( ColReduceSum( SelectFirst( Accum, ColReduceSumSq( Square(Accum) ) ) ) )
    // ------------------------------------------------------------------
    typename EVT_Y::Arguments callback_args{
        // EVT_RowReduceSum: { child0 = EVT_DropSq, sum-reduce args }
        {
            // EVT_DropSq: { child0 = Accum, child1 = EVT_RowReduceSq, op = SelectFirst }
            {
                {},   // Accum (no args)
                // EVT_RowReduceSq: { child0 = EVT_Square, sumsq-reduce args }
                {
                    // EVT_Square: { child0 = Accum, op = ComputeSquare }
                    {
                        {},   // Accum
                        {}    // ComputeSquare
                    },
                    // ColReduceSumSq args: pointer, identity, stride
                    {ptrSumSq, ElementReduce(0), {_1{}, _0{}, _0{}}}
                },
                {}    // ComputeSelectFirst (no args)
            },
            // ColReduceSum args: pointer, identity, stride
            {ptrSum,   ElementReduce(0), {_1{}, _0{}, _0{}}}
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
                "xnorm GEMM can_implement failed: ",
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
                "xnorm GEMM init failed: ", cutlassGetStatusString(st));
    st = op(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "xnorm GEMM run failed: ", cutlassGetStatusString(st));

    // ------------------------------------------------------------------
    // Kernel 2: finalize (per-norm-type) → D
    // ------------------------------------------------------------------
    float const inv_N = 1.0f / static_cast<float>(N);
    dim3 const block(kFinalizeBlockN);
    dim3 const grid((N + kFinalizeBlockN - 1) / kFinalizeBlockN, M);

    auto* Yp = reinterpret_cast<__nv_bfloat16 const*>(ptrY);
    switch (norm_kind) {
        case NormType::RMSNorm:
            xnorm_finalize_kernel<NormType::RMSNorm><<<grid, block, 0, stream>>>(
                Yp, ptrGamma, ptrBeta, ptrSum, ptrSumSq, ptrD,
                M, N, static_cast<float>(eps), inv_N);
            break;
        case NormType::LayerNorm:
            xnorm_finalize_kernel<NormType::LayerNorm><<<grid, block, 0, stream>>>(
                Yp, ptrGamma, ptrBeta, ptrSum, ptrSumSq, ptrD,
                M, N, static_cast<float>(eps), inv_N);
            break;
        case NormType::L2Norm:
            xnorm_finalize_kernel<NormType::L2Norm><<<grid, block, 0, stream>>>(
                Yp, ptrGamma, ptrBeta, ptrSum, ptrSumSq, ptrD,
                M, N, static_cast<float>(eps), inv_N);
            break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUTLASS fused matmul + {RMSNorm, LayerNorm, L2Norm} (bf16) on sm_120";
    m.def("xnorm_matmul_out",
          &xnorm_matmul_out,
          "D = norm(A @ B.T) * gamma [+ beta]; "
          "norm_type ∈ {rmsnorm, layernorm, l2norm}; "
          "A:(M,K) bf16, B:(N,K) bf16, gamma:(N,) bf16 or None, "
          "beta:(N,) bf16 or None (layernorm only), D:(M,N) bf16",
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("gamma") = pybind11::none(),
          pybind11::arg("beta")  = pybind11::none(),
          pybind11::arg("D"),
          pybind11::arg("norm_type"),
          pybind11::arg("eps") = 1e-6);
}
