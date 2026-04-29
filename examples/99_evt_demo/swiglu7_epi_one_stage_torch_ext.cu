// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Single-kernel fully-fused swiglu7:
//
//   D = swiglu7(A @ B.T)
//
//   A : (M, K)   bf16 row-major
//   B : (N, K)   bf16 row-major   (torch.nn.Linear weight convention; N even)
//   D : (M, N/2) bf16 row-major
//
// Python definition this kernel implements exactly:
//
//   def swiglu7(x, alpha=1.702, limit=7.0):
//       x = x.to(float32)
//       x_glu, x_linear = x[..., 0::2], x[..., 1::2]
//       x_glu    = x_glu.clamp(max=limit)
//       x_linear = x_linear.clamp(min=-limit, max=limit)
//       out_glu  = x_glu * sigmoid(alpha * x_glu)
//       return out_glu * (x_linear + 1.0)
//
// Implementation: a single CUTLASS kernel call via
// cutlass::gemm::device::DualGemm (sm_80 multistage; runs on sm_120 via
// backward compatibility). The two GEMMs A @ W_gate.T and A @ W_linear.T
// run in the same threadblock sharing A's smem stages; their accumulators
// stay in registers and a custom Swiglu7Combine epilogue functor combines
// them and writes only D. With kStoreD0 = kStoreD1 = false, neither
// intermediate (M, N/2) fragment touches HBM — eliminating the
// linear_buf round-trip that the two-stage variant pays.
//
// W_gate and W_linear are accessed as strided ColumnMajor views of B
// (ldB = 2K) — even rows for gate (base ptrB), odd rows for linear
// (base ptrB + K). Zero copy.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/util/host_tensor.h"

#include "../45_dual_gemm/device/dual_gemm.h"
#include "swiglu7_combine.h"

////////////////////////////////////////////////////////////////////////////////
// Data types
////////////////////////////////////////////////////////////////////////////////

using ElementA       = cutlass::bfloat16_t;
using ElementB       = cutlass::bfloat16_t;
using ElementC       = cutlass::bfloat16_t;
using ElementAcc     = float;
using ElementCompute = float;

using LayoutA  = cutlass::layout::RowMajor;
using LayoutB0 = cutlass::layout::ColumnMajor;   // strided ldB = 2K view
using LayoutB1 = cutlass::layout::ColumnMajor;   // strided ldB = 2K view
using LayoutC  = cutlass::layout::RowMajor;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // = 8
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // = 8
// Output vector width carried by the epilogue ops. 4 (not 8) so that any
// (M, N/2) output whose row stride is a multiple of 4 bf16 elements (8 bytes)
// is supported — N values like 27304 produce N/2 = 13652 which is not
// 8-aligned but is 4-aligned.
constexpr int EpilogueVecCount = 4;

////////////////////////////////////////////////////////////////////////////////
// Tile shapes
//
// Smaller than the EVT version (128x128x32 / 4 stages) because DualGemm
// stages two B operands in smem simultaneously. The example
// examples/45_dual_gemm/dual_gemm.cu uses these exact dimensions for
// the bf16 / fp16 sm80 fused configuration.
////////////////////////////////////////////////////////////////////////////////

using ArchTag          = cutlass::arch::Sm80;
using OperatorClass    = cutlass::arch::OpClassTensorOp;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
using WarpShape        = cutlass::gemm::GemmShape< 64, 32, 32>;
using InstructionShape = cutlass::gemm::GemmShape< 16,  8, 16>;
constexpr int kStages         = 3;
constexpr bool kSplitKSerial  = false;
constexpr bool kStoreD0       = false;   // gate accumulator never spills to HBM
constexpr bool kStoreD1       = false;   // linear accumulator never spills to HBM

////////////////////////////////////////////////////////////////////////////////
// Three epilogue operators
//
// Op0 / Op1 are the per-GEMM linear combinations. With kStoreD0/D1 = false
// they're not actually written, but the kernel still instantiates the type;
// ScaleType::Nothing makes the linear-combine itself a no-op.
//
// Op2 is the binary swiglu7 combine that consumes both accumulators and
// produces D in registers, then stores D via the standard epilogue path.
////////////////////////////////////////////////////////////////////////////////

constexpr auto kScaleType = cutlass::epilogue::thread::ScaleType::Nothing;

using EpilogueOp0 = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    EpilogueVecCount,
    ElementAcc,
    ElementCompute,
    kScaleType>;

using EpilogueOp1 = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    EpilogueVecCount,
    ElementAcc,
    ElementCompute,
    kScaleType>;

using EpilogueOp2 = cutlass::epilogue::thread::Swiglu7Combine<
    ElementC,
    EpilogueVecCount,
    ElementAcc,
    ElementCompute>;

////////////////////////////////////////////////////////////////////////////////
// DualGemm device type
////////////////////////////////////////////////////////////////////////////////

using DualGemm = cutlass::gemm::device::DualGemm<
    ElementA, LayoutA,
    ElementB,
    LayoutB0, LayoutB1,
    ElementC, LayoutC,
    ElementAcc,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp0,
    EpilogueOp1,
    EpilogueOp2,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    kStages,
    kStoreD0,
    kStoreD1,
    kSplitKSerial,
    AlignmentA,
    AlignmentB>;

////////////////////////////////////////////////////////////////////////////////
// Python-facing launcher
////////////////////////////////////////////////////////////////////////////////

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
    int const N = static_cast<int>(B.size(0));

    TORCH_CHECK((N % 2) == 0, "N (= B.size(0)) must be even, got ", N);
    int const N_out = N / 2;

    TORCH_CHECK(D.size(0) == M && D.size(1) == N_out,
                "D must be (M, N/2) = (", M, ",", N_out, "), got ", D.sizes());

    auto ptrA = reinterpret_cast<ElementA*>(A.data_ptr<at::BFloat16>());
    auto ptrB = reinterpret_cast<ElementB*>(B.data_ptr<at::BFloat16>());
    auto ptrD = reinterpret_cast<ElementC*>(D.data_ptr<at::BFloat16>());

    int64_t const ldB_strided = static_cast<int64_t>(2) * K;

    // ColumnMajor layout objects holding the strided ldB.
    LayoutB0 layoutB_gate  (ldB_strided);
    LayoutB1 layoutB_linear(ldB_strided);
    LayoutC  layoutC       (static_cast<int64_t>(N_out));

    // TensorRefs. ref_C0/C1 and ref_D0/D1 are not dereferenced because
    // ScaleType::Nothing skips bias and kStoreD0/D1=false skips writes,
    // but we still pass valid layouts (stride 0) with nullptr data.
    using TensorRefA  = cutlass::TensorRef<ElementA const, LayoutA>;
    using TensorRefB0 = cutlass::TensorRef<ElementB const, LayoutB0>;
    using TensorRefB1 = cutlass::TensorRef<ElementB const, LayoutB1>;
    using TensorRefCi = cutlass::TensorRef<ElementC const, LayoutC>;
    using TensorRefDo = cutlass::TensorRef<ElementC,       LayoutC>;

    TensorRefA  ref_A0(ptrA,        LayoutA(static_cast<int64_t>(K)));
    TensorRefB0 ref_B0(ptrB,        layoutB_gate);                 // W_gate (even rows)
    TensorRefCi ref_C0(nullptr,     LayoutC(0));
    TensorRefDo ref_D0(nullptr,     LayoutC(0));
    TensorRefB1 ref_B1(ptrB + K,    layoutB_linear);               // W_linear (odd rows)
    TensorRefCi ref_C1(nullptr,     LayoutC(0));
    TensorRefDo ref_D1(nullptr,     LayoutC(0));
    TensorRefDo ref_D2(ptrD,        layoutC);                      // output

    typename EpilogueOp0::Params epi0{ElementCompute(1.0f), ElementCompute(0.0f)};
    typename EpilogueOp1::Params epi1{ElementCompute(1.0f), ElementCompute(0.0f)};
    typename EpilogueOp2::Params epi2{};

    cutlass::gemm::GemmCoord problem{M, N_out, K};

    typename DualGemm::Arguments args(
        cutlass::gemm::DualGemmMode::kGemm,
        problem,
        ref_A0,
        ref_B0, ref_C0, ref_D0,
        ref_B1, ref_C1, ref_D1,
        ref_D2,
        epi0, epi1, epi2,
        /*split_k_slices=*/1,
        /*batch_count=*/1,
        /*batch_stride_A=*/0,
        /*batch_stride_B0=*/0,
        /*batch_stride_B1=*/0,
        /*batch_stride_C=*/0,
        /*batch_stride_D=*/0);

    DualGemm op;

    cutlass::Status st = op.can_implement(args);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "DualGemm can_implement failed: ",
                cutlassGetStatusString(st),
                " for (M,N_out,K)=(", M, ",", N_out, ",", K, ")");

    size_t workspace_bytes = DualGemm::get_workspace_size(args);
    at::Tensor workspace;
    void* workspace_ptr = nullptr;
    if (workspace_bytes > 0) {
        workspace = at::empty({static_cast<int64_t>(workspace_bytes)},
                              at::TensorOptions().dtype(at::kByte).device(A.device()));
        workspace_ptr = workspace.data_ptr();
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index()).stream();

    st = op.initialize(args, workspace_ptr, stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "DualGemm init failed: ", cutlassGetStatusString(st));
    st = op(stream);
    TORCH_CHECK(st == cutlass::Status::kSuccess,
                "DualGemm run failed: ", cutlassGetStatusString(st));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUTLASS DualGemm fully-fused swiglu7 (bf16) on sm_120";
    m.def("swiglu7_matmul_out",
          &swiglu7_matmul_out,
          "D = swiglu7(A @ B.T) in a single fused kernel; "
          "A:(M,K) bf16, B:(N,K) bf16 (N even), D:(M,N/2) bf16",
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("D"));
}
