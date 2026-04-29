// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Python-binding for a CUTLASS implementation of the canonical swiglu7
// projection on Blackwell Geforce (sm_120, RTX 5090):
//
//   D = swiglu7(A @ B.T)
//
//   A : (M, K)   bf16 row-major   -- activations
//   B : (N, K)   bf16 row-major   -- combined gate+linear weight (nn.Linear conv.)
//                                    N must be even; even rows are gate weights,
//                                    odd rows are linear weights (interleaved)
//   D : (M, N/2) bf16 row-major   -- output
//
// where, in pure PyTorch:
//
//   def swiglu7(x, alpha=1.702, limit=7.0):
//       x = x.to(float32)
//       x_glu, x_linear = x[..., 0::2], x[..., 1::2]
//       x_glu    = x_glu.clamp(max=limit)
//       x_linear = x_linear.clamp(min=-limit, max=limit)
//       out_glu  = x_glu * sigmoid(alpha * x_glu)
//       return out_glu * (x_linear + 1.0)
//
// Implementation: a single launcher fires two back-to-back Sm80 bf16 GEMMs
// on the current CUDA stream, sharing the same B buffer through strided
// views (ldB = 2K) of its even (gate) and odd (linear) rows:
//
//   Stage 1 (vanilla GEMM, LinearCombination epilogue):
//       linear_buf = A @ W_linear.T,      W_linear = B[1::2, :]   (M, N/2) bf16
//
//   Stage 2 (EVT GEMM, fused swiglu7 epilogue):
//       D = SiLU_alpha(clamp(A @ W_gate.T, max=7))
//             * (clamp(linear_buf, -7, 7) + 1),   W_gate = B[0::2, :]
//
// EVT tree (Stage 2, bottom-up):
//   T0 = clamp(Accum, max=limit)           (ComputeClampMax7, unary)
//   T1 = SiLU_alpha(T0)                    (ComputeSiLuAlpha, x*sigmoid(alpha*x))
//   T2 = linear_buf                        (VisitorAuxLoad from Stage 1 scratch)
//   T3 = clamp(T2, -limit, limit)          (ComputeClampSymm7, unary)
//   T4 = T3 + 1.0                          (ComputePlusOne, unary)
//   T5 = T1 * T4                           (ComputeMul, binary)
//   D  = AuxStore(T5)
//
// Both GEMMs are Ampere (Sm80) TensorOp MMAs that compile and run on
// sm_120 via backward compatibility. Accumulation and epilogue compute
// are fp32.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/functional.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"

#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

using cute::_0;
using cute::_1;

////////////////////////////////////////////////////////////////////////////////
// Custom epilogue functors for swiglu7
// Each follows the CUTLASS pattern:
//   primary template:  template <typename T> struct Foo        -- scalar
//   partial spec:      template <typename T, int N>
//                      struct Foo<cutlass::Array<T,N>>         -- vector
////////////////////////////////////////////////////////////////////////////////

// clamp(x, max=7)  -- only upper-clamps the gate accumulator
template <typename T>
struct ClampMax7 {
    CUTLASS_HOST_DEVICE
    T operator()(T const& x) const {
        return x < T(7.0f) ? x : T(7.0f);
    }
};

template <typename T, int N>
struct ClampMax7<cutlass::Array<T, N>> {
    CUTLASS_HOST_DEVICE
    cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& x) const {
        ClampMax7<T> op;
        cutlass::Array<T, N> out;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) out[i] = op(x[i]);
        return out;
    }
};

// x * sigmoid(1.702 * x)  -- scaled SiLU used in swiglu7
template <typename T>
struct SiLuAlpha1702 {
    static const bool kIsHeavy = true;
    CUTLASS_HOST_DEVICE
    T operator()(T const& x) const {
        cutlass::epilogue::thread::Sigmoid<T> sig;
        return x * sig(T(1.702f) * x);
    }
};

template <typename T, int N>
struct SiLuAlpha1702<cutlass::Array<T, N>> {
    static const bool kIsHeavy = true;
    CUTLASS_HOST_DEVICE
    cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& x) const {
        SiLuAlpha1702<T> op;
        cutlass::Array<T, N> out;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) out[i] = op(x[i]);
        return out;
    }
};

// clamp(x, -7, 7)  -- symmetric clamp applied to the linear path
template <typename T>
struct ClampSymm7 {
    CUTLASS_HOST_DEVICE
    T operator()(T const& x) const {
        T lo(-7.0f), hi(7.0f);
        return x < lo ? lo : (x > hi ? hi : x);
    }
};

template <typename T, int N>
struct ClampSymm7<cutlass::Array<T, N>> {
    CUTLASS_HOST_DEVICE
    cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& x) const {
        ClampSymm7<T> op;
        cutlass::Array<T, N> out;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) out[i] = op(x[i]);
        return out;
    }
};

// x + 1.0
template <typename T>
struct PlusOne {
    CUTLASS_HOST_DEVICE
    T operator()(T const& x) const { return x + T(1.0f); }
};

template <typename T, int N>
struct PlusOne<cutlass::Array<T, N>> {
    CUTLASS_HOST_DEVICE
    cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& x) const {
        PlusOne<T> op;
        cutlass::Array<T, N> out;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) out[i] = op(x[i]);
        return out;
    }
};

////////////////////////////////////////////////////////////////////////////////
// Data types and layouts
////////////////////////////////////////////////////////////////////////////////

using ElementA       = cutlass::bfloat16_t;
using ElementB       = cutlass::bfloat16_t;
using ElementC       = cutlass::bfloat16_t;  // linear / D type
using ElementAcc     = float;               // tensor-core accumulator
using ElementCompute = float;               // epilogue compute precision

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;  // (N, K) row-major == (K, N) col-major
using LayoutC = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // = 8
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
// AlignmentC = 4 (not 8) so that any (M, N/2) output whose row stride is a
// multiple of 4 bf16 elements (8 bytes) is supported. With AlignmentC = 8
// (16-byte stores) shapes like N = 27304 fail because N/2 = 13652 is not a
// multiple of 8. The vectorization cost is negligible for compute-bound
// shapes; fully aligned shapes still benefit from it through the inner loops.
constexpr int AlignmentC = 4;

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
//
//  Accum
//    └─ ClampMax7 ─── SiLuAlpha ─────────────────── ComputeMul ─── StoreD → D
//                                                        │
//  LinearLoad ─── ClampSymm7 ─── PlusOne ───────────────┘
////////////////////////////////////////////////////////////////////////////////

using OutputTileThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
    ThreadblockShape, WarpShape, ElementC, AlignmentC, EVTEpilogueStages>;

using Accum = cutlass::epilogue::threadblock::VisitorAccFetch;

// Linear path: load (M,N) auxiliary matrix
using LinearLoad = cutlass::epilogue::threadblock::VisitorAuxLoad<
    OutputTileThreadMap, ElementC,
    cute::Stride<int64_t, _1, int64_t>>;

// Binary multiply for gate * linear
using ComputeMul = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, ElementCompute, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

// Gate path custom ops
using ComputeClampMax7 = cutlass::epilogue::threadblock::VisitorCompute<
    ClampMax7, ElementCompute, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using ComputeSiLuAlpha = cutlass::epilogue::threadblock::VisitorCompute<
    SiLuAlpha1702, ElementCompute, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

// Linear path custom ops
using ComputeClampSymm7 = cutlass::epilogue::threadblock::VisitorCompute<
    ClampSymm7, ElementCompute, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using ComputePlusOne = cutlass::epilogue::threadblock::VisitorCompute<
    PlusOne, ElementCompute, ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

// Output store: (M,N) row-major
using StoreD = cutlass::epilogue::threadblock::VisitorAuxStore<
    OutputTileThreadMap, ElementC,
    cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<int64_t, _1, int64_t>>;

// --- Gate branch ---
// T0 = clamp(Accum, max=7)
using EVT_ClampGate = cutlass::epilogue::threadblock::Sm80EVT<
    ComputeClampMax7, Accum>;

// T1 = SiLU_alpha(T0) = T0 * sigmoid(1.702 * T0)
using EVT_SiLuGate = cutlass::epilogue::threadblock::Sm80EVT<
    ComputeSiLuAlpha, EVT_ClampGate>;

// --- Linear branch ---
// T2 = LinearLoad
// T3 = clamp(T2, -7, 7)
using EVT_ClampLin = cutlass::epilogue::threadblock::Sm80EVT<
    ComputeClampSymm7, LinearLoad>;

// T4 = T3 + 1.0
using EVT_PlusOne = cutlass::epilogue::threadblock::Sm80EVT<
    ComputePlusOne, EVT_ClampLin>;

// --- Combine ---
// T5 = T1 * T4
using EVT_Mul = cutlass::epilogue::threadblock::Sm80EVT<
    ComputeMul, EVT_SiLuGate, EVT_PlusOne>;

// D = store(T5)
using EVT_D = cutlass::epilogue::threadblock::Sm80EVT<
    StoreD, EVT_Mul>;

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
// Stage-1 vanilla Sm80 bf16 GEMM (LinearCombination epilogue)
//
//   linear_buf = A @ W_linear.T,  W_linear = B[1::2, :]  -> (M, N/2) bf16
//
// Same tile/warp/instruction shapes and operand layouts as the EVT GEMM,
// so the two stages share kernel code paths and tile geometry.
////////////////////////////////////////////////////////////////////////////////

using VanillaEpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    AlignmentC,
    ElementAcc,
    ElementCompute>;

using VanillaGemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAcc,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    VanillaEpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages>;

////////////////////////////////////////////////////////////////////////////////
// Python-facing launcher
////////////////////////////////////////////////////////////////////////////////

// D = swiglu7(A @ B.T)  with B = interleaved [W_gate, W_linear] along rows.
//   A : (M, K)   bf16 row-major
//   B : (N, K)   bf16 row-major   (torch.nn.Linear weight convention; N even)
//   D : (M, N/2) bf16 row-major
void swiglu7_matmul_out(at::Tensor A,
                        at::Tensor B,
                        at::Tensor D) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && D.is_cuda(),
                "all inputs must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16
                    && D.scalar_type() == at::kBFloat16,
                "all inputs must be bf16 (torch.bfloat16)");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && D.dim() == 2, "A, B, D must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1),
                "K mismatch: A.size(1) must equal B.size(1) (B is (N,K))");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && D.is_contiguous(),
                "A, B, D must be contiguous (row-major)");

    int const M = static_cast<int>(A.size(0));
    int const K = static_cast<int>(A.size(1));
    int const N = static_cast<int>(B.size(0));   // B is (N, K)

    TORCH_CHECK((N % 2) == 0, "N (= B.size(0)) must be even, got ", N);
    int const N_out = N / 2;

    TORCH_CHECK(D.size(0) == M && D.size(1) == N_out,
                "D must be (M, N/2) = (", M, ",", N_out, "), got ",
                D.sizes());

    auto ptrA = reinterpret_cast<ElementA*>(A.data_ptr<at::BFloat16>());
    auto ptrB = reinterpret_cast<ElementB*>(B.data_ptr<at::BFloat16>());
    auto ptrD = reinterpret_cast<ElementC*>(D.data_ptr<at::BFloat16>());

    // Strided views of B (row-major (N,K)) as ColumnMajor (K, N_out) with ld 2K:
    //   W_gate   : even rows of B, base = ptrB
    //   W_linear : odd  rows of B, base = ptrB + K
    int64_t const ldB_strided = static_cast<int64_t>(2) * K;
    auto ptrB_gate   = ptrB;
    auto ptrB_linear = ptrB + K;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();

    // Scratch buffer for the linear path output (M, N/2) bf16, row-major.
    at::Tensor linear_buf = at::empty({M, N_out}, A.options());
    auto ptrLinear = reinterpret_cast<ElementC*>(linear_buf.data_ptr<at::BFloat16>());

    cutlass::gemm::GemmCoord problem{M, N_out, K};

    // ------------------------------------------------------------------
    // Stage 1: linear_buf = A @ W_linear.T   (vanilla bf16 GEMM)
    // ------------------------------------------------------------------
    {
        typename VanillaGemm::Arguments args{
            problem,
            {ptrA,        static_cast<int64_t>(K)},          // A  : ld K
            {ptrB_linear, ldB_strided},                      // B  : ld 2K, odd rows
            {ptrLinear,   static_cast<int64_t>(N_out)},      // C  : unused (beta=0)
            {ptrLinear,   static_cast<int64_t>(N_out)},      // D  : ld N_out
            {ElementAcc(1.0f), ElementAcc(0.0f)},
            /*split_k_slices=*/1};

        VanillaGemm op;
        cutlass::Status st = op.initialize(args, /*workspace=*/nullptr, stream);
        TORCH_CHECK(st == cutlass::Status::kSuccess,
                    "CUTLASS stage-1 init failed: ", cutlassGetStatusString(st),
                    " for (M,N_out,K)=(", M, ",", N_out, ",", K, ")");
        st = op(stream);
        TORCH_CHECK(st == cutlass::Status::kSuccess,
                    "CUTLASS stage-1 run failed: ", cutlassGetStatusString(st));
    }

    // ------------------------------------------------------------------
    // Stage 2: D = SiLU_alpha(clamp(A @ W_gate.T, max=7))
    //              * (clamp(linear_buf, -7, 7) + 1)        (EVT GEMM)
    // ------------------------------------------------------------------
    int64_t const MN_out = static_cast<int64_t>(M) * static_cast<int64_t>(N_out);

    typename EVT_D::Arguments callback_args{
        // EVT_Mul: {child0=EVT_SiLuGate, child1=EVT_PlusOne, op=ComputeMul}
        {
            // EVT_SiLuGate: {child0=EVT_ClampGate, op=ComputeSiLuAlpha}
            {
                // EVT_ClampGate: {child0=Accum, op=ComputeClampMax7}
                {
                    {},   // Accum (no args)
                    {}    // ComputeClampMax7 (no args)
                },
                {}        // ComputeSiLuAlpha (no args)
            },
            // EVT_PlusOne: {child0=EVT_ClampLin, op=ComputePlusOne}
            {
                // EVT_ClampLin: {child0=LinearLoad, op=ComputeClampSymm7}
                {
                    {ptrLinear, ElementC(0), {int64_t(N_out), _1{}, MN_out}},
                    {}    // ComputeClampSymm7 (no args)
                },
                {}        // ComputePlusOne (no args)
            },
            {}            // ComputeMul (no args)
        },
        {ptrD, {int64_t(N_out), _1{}, MN_out}}  // StoreD
    };

    typename DeviceGemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        /*batch_count=*/1,
        callback_args,
        ptrA,
        ptrB_gate,
        /*ptr_C=*/nullptr,
        /*ptr_D=*/nullptr,
        /*batch_stride_A=*/static_cast<int64_t>(M) * K,
        /*batch_stride_B=*/static_cast<int64_t>(N) * K,
        /*batch_stride_C=*/0,
        /*batch_stride_D=*/0,
        /*stride_a=*/static_cast<int64_t>(K),  // ldA (RowMajor M x K)
        /*stride_b=*/ldB_strided,              // ldB (strided view of B, even rows)
        /*stride_c=*/0,
        /*stride_d=*/0);

    DeviceGemm op;

    cutlass::Status st = op.can_implement(args);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "CUTLASS stage-2 can_implement failed: ",
                cutlassGetStatusString(st),
                " for (M,N_out,K)=(", M, ",", N_out, ",", K, ")");

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
                "CUTLASS stage-2 init failed: ", cutlassGetStatusString(st));
    st = op(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "CUTLASS stage-2 run failed: ", cutlassGetStatusString(st));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUTLASS two-stage GEMM with fused swiglu7 epilogue (bf16) on sm_120";
    m.def("swiglu7_matmul_out",
          &swiglu7_matmul_out,
          "D = swiglu7(A @ B.T); "
          "A:(M,K) bf16, B:(N,K) bf16 (N even), D:(M,N/2) bf16",
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("D"));
}
