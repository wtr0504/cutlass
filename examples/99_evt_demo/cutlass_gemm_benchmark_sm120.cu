// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

//
// Benchmark: CUTLASS 2.x fp16 TensorOp GEMM vs cuBLAS on RTX 5090 (sm_120).
//
// torch.mm(fp16) on sm_120 dispatches to cuBLAS (cuBLASLt). To keep this
// benchmark self-contained and libtorch-free we call cuBLAS directly via
// cublasGemmEx, which is functionally equivalent and what PyTorch calls into.
//
// CUTLASS 3.x CollectiveBuilder on sm_120 currently only supports F8F6F4 MMA
// (see include/cutlass/gemm/collective/builders/sm120_mma_builder.inl:83),
// so the CUTLASS side here uses the CUTLASS 2.x ``cutlass::gemm::device::Gemm``
// with Sm80 TensorOp (``mma.m16n8k16.f16``), which compiles and runs fine on
// Blackwell Geforce.
//
// Layout convention (mirrors the CuTeDSL benchmark):
//   - A is (M,K) row-major
//   - B is (N,K) row-major (i.e. we interpret it as column-major (K,N))
//   - C is (M,N) row-major
//   -> compute C = A * B^T, matching torch.mm(A, B.t()).
//
// Build:
//   /usr/local/cuda/bin/nvcc -std=c++17 -O3                               \
//     -gencode arch=compute_120,code=sm_120                               \
//     --expt-relaxed-constexpr                                            \
//     -I/root/cutlass/include -I/root/cutlass/tools/util/include          \
//     cutlass_gemm_benchmark_sm120.cu -o cutlass_gemm_benchmark_sm120     \
//     -lcublas
//
// See build_cutlass_gemm_benchmark_sm120.sh in the same directory.
//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                   cudaGetErrorString(err));                                    \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#define CUBLAS_CHECK(call)                                                      \
  do {                                                                          \
    cublasStatus_t _st = (call);                                                \
    if (_st != CUBLAS_STATUS_SUCCESS) {                                         \
      std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,      \
                   static_cast<int>(_st));                                      \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

using ElementA   = cutlass::half_t;
using ElementB   = cutlass::half_t;
using ElementC   = cutlass::half_t;
using ElementAcc = float;

using LayoutA = cutlass::layout::RowMajor;      // A: (M,K) row-major
using LayoutB = cutlass::layout::ColumnMajor;   // B: (K,N) col-major == (N,K) row-major
using LayoutC = cutlass::layout::RowMajor;      // C: (M,N) row-major

using MMAOp   = cutlass::arch::OpClassTensorOp;
using SmArch  = cutlass::arch::Sm80;  // Ampere tensor-op m16n8k16.f16, runs on sm_120

// Reasonably fast general-purpose configuration.
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

struct Shape { int M, N, K; };

static std::vector<Shape> DEFAULT_SHAPES = {
    {4096,  4096,  4096},
    {4096,  8192,  4096},
    {8192,  8192,  4096},
    {2048,  8192,  8192},
    {16384, 4096,  4096},
    {1024,  14336, 4096},
    {4096,  14336, 4096},
};

static std::vector<Shape> parse_shapes(std::string const& s) {
  std::vector<Shape> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ';')) {
    if (tok.empty()) continue;
    int M, N, K;
    if (std::sscanf(tok.c_str(), "%d,%d,%d", &M, &N, &K) == 3) {
      out.push_back({M, N, K});
    }
  }
  return out;
}

static double tflops(double ms, int M, int N, int K) {
  return 2.0 * double(M) * double(N) * double(K) / (ms * 1e-3) / 1e12;
}

// Runs ``launch(stream)`` ``iters`` times on ``stream`` and returns avg ms.
template <typename LaunchFn>
static double bench(LaunchFn launch, cudaStream_t stream,
                    int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) launch(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(s, stream));
  for (int i = 0; i < iters; ++i) launch(stream);
  CUDA_CHECK(cudaEventRecord(e, stream));
  CUDA_CHECK(cudaEventSynchronize(e));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
  CUDA_CHECK(cudaEventDestroy(s));
  CUDA_CHECK(cudaEventDestroy(e));
  return double(ms) / double(iters);
}

// Launch CUTLASS GEMM on ``stream``. A/B/C are device fp16 pointers.
static void launch_cutlass_gemm(Gemm& op, cudaStream_t stream,
                                int M, int N, int K,
                                cutlass::half_t* A_dev,
                                cutlass::half_t* B_dev,
                                cutlass::half_t* C_dev) {
  cutlass::gemm::GemmCoord problem{M, N, K};
  typename Gemm::Arguments args{
      problem,
      {A_dev, K},   // A: RowMajor lda = K
      {B_dev, K},   // B: ColumnMajor ldb = K (same memory as (N,K) row-major)
      {C_dev, N},   // C: RowMajor ldc = N
      {C_dev, N},
      {ElementAcc(1.0f), ElementAcc(0.0f)},
      /*split_k_slices=*/1};
  cutlass::Status status = op.initialize(args, /*workspace=*/nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "CUTLASS init failed for (%d,%d,%d): %s\n",
                 M, N, K, cutlassGetStatusString(status));
    std::exit(1);
  }
  status = op(stream);
  if (status != cutlass::Status::kSuccess) {
    std::fprintf(stderr, "CUTLASS run failed for (%d,%d,%d): %s\n",
                 M, N, K, cutlassGetStatusString(status));
    std::exit(1);
  }
}

// cuBLAS baseline: compute row-major C(M,N) = A(M,K) * B(N,K)^T in fp16 with
// fp32 accumulation. cuBLAS is column-major so we compute
//   C^T(N,M) = B(N,K) * A(M,K)^T
// by telling cuBLAS:
//   opA = OP_T applied to the B pointer (col-major (K,N) -> op_T gives (N,K) = math B)
//   opB = OP_N applied to the A pointer (col-major (K,M) = math A^T)
//   m=N, n=M, k=K, lda_B=K, lda_A=K, ldc=N.
static void launch_cublas_gemm(cublasHandle_t handle,
                               int M, int N, int K,
                               __half const* A_dev,
                               __half const* B_dev,
                               __half*       C_dev) {
  float const alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(
      handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      /*m=*/N, /*n=*/M, /*k=*/K,
      &alpha,
      B_dev, CUDA_R_16F, /*ldb->lda=*/K,
      A_dev, CUDA_R_16F, /*lda->ldb=*/K,
      &beta,
      C_dev, CUDA_R_16F, /*ldc=*/N,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void fill_random_half(cutlass::half_t* host_ptr, size_t n,
                             int seed, float lo, float hi) {
  std::srand(seed);
  for (size_t i = 0; i < n; ++i) {
    float v = lo + (hi - lo) * (float(std::rand()) / float(RAND_MAX));
    host_ptr[i] = cutlass::half_t(v);
  }
}

struct Buffers {
  cutlass::half_t* dA = nullptr;
  cutlass::half_t* dB = nullptr;
  cutlass::half_t* dC_cutlass = nullptr;
  cutlass::half_t* dC_cublas  = nullptr;

  void alloc(int M, int N, int K) {
    free_all();
    size_t aN = size_t(M) * size_t(K);
    size_t bN = size_t(N) * size_t(K);
    size_t cN = size_t(M) * size_t(N);
    CUDA_CHECK(cudaMalloc(&dA, aN * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&dB, bN * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&dC_cutlass, cN * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&dC_cublas,  cN * sizeof(cutlass::half_t)));

    std::vector<cutlass::half_t> hA(aN), hB(bN);
    fill_random_half(hA.data(), aN, 7,  -1.0f, 1.0f);
    fill_random_half(hB.data(), bN, 13, -1.0f, 1.0f);
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), aN * sizeof(cutlass::half_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bN * sizeof(cutlass::half_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC_cutlass, 0, cN * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMemset(dC_cublas,  0, cN * sizeof(cutlass::half_t)));
  }
  void free_all() {
    if (dA) cudaFree(dA); dA = nullptr;
    if (dB) cudaFree(dB); dB = nullptr;
    if (dC_cutlass) cudaFree(dC_cutlass); dC_cutlass = nullptr;
    if (dC_cublas)  cudaFree(dC_cublas);  dC_cublas  = nullptr;
  }
  ~Buffers() { free_all(); }
};

// Compare CUTLASS and cuBLAS outputs to sanity-check the layout setup.
static void validate(Buffers const& buf, int M, int N) {
  size_t cN = size_t(M) * size_t(N);
  std::vector<cutlass::half_t> h_cutlass(cN), h_cublas(cN);
  CUDA_CHECK(cudaMemcpy(h_cutlass.data(), buf.dC_cutlass,
                        cN * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_cublas.data(), buf.dC_cublas,
                        cN * sizeof(cutlass::half_t), cudaMemcpyDeviceToHost));

  double max_abs_diff = 0.0;
  double sum_abs_cublas = 0.0;
  for (size_t i = 0; i < cN; ++i) {
    double a = double(float(h_cutlass[i]));
    double b = double(float(h_cublas[i]));
    max_abs_diff = std::max(max_abs_diff, std::fabs(a - b));
    sum_abs_cublas += std::fabs(b);
  }
  double mean_cublas = sum_abs_cublas / double(cN);
  double rel = max_abs_diff / std::max(1e-6, mean_cublas);
  std::cout << "Correctness (" << M << "," << N
            << "): max|cutlass-cublas|=" << std::scientific
            << std::setprecision(2) << max_abs_diff
            << ", rel=" << rel << std::defaultfloat << "\n";
}

int main(int argc, char** argv) {
  std::string shapes_str;
  int iters  = 20;
  int warmup = 5;
  bool skip_validate = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(1); }
      return argv[++i];
    };
    if (a == "--shapes") shapes_str = next();
    else if (a == "--iters") iters = std::stoi(next());
    else if (a == "--warmup") warmup = std::stoi(next());
    else if (a == "--skip-validate") skip_validate = true;
    else if (a == "--help" || a == "-h") {
      std::cout << "Usage: " << argv[0]
                << " [--shapes M,N,K;M,N,K;...] [--iters N] [--warmup N] [--skip-validate]\n";
      return 0;
    }
  }

  auto shapes = shapes_str.empty() ? DEFAULT_SHAPES : parse_shapes(shapes_str);
  if (shapes.empty()) { std::cerr << "no shapes\n"; return 1; }

  int dev = 0;
  CUDA_CHECK(cudaSetDevice(dev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  std::cout << "Device: " << prop.name
            << " (sm_" << prop.major << prop.minor << ")\n";

  // Single stream shared by both implementations so timings are apples-to-apples.
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cublasHandle_t cublas;
  CUBLAS_CHECK(cublasCreate(&cublas));
  CUBLAS_CHECK(cublasSetStream(cublas, stream));
  CUBLAS_CHECK(cublasSetMathMode(cublas, CUBLAS_TENSOR_OP_MATH));

  Gemm cutlass_op;

  std::cout << "\n"
            << std::left << std::setw(22) << "shape (M,N,K)"
            << std::right << std::setw(12) << "cuBLAS ms"
            << std::setw(12) << "cuBLAS TF"
            << std::setw(13) << "CUTLASS ms"
            << std::setw(13) << "CUTLASS TF"
            << std::setw(10) << "speedup"
            << "\n";
  std::cout << std::string(82, '-') << "\n";

  for (Shape const& sh : shapes) {
    int const M = sh.M, N = sh.N, K = sh.K;
    Buffers buf;
    buf.alloc(M, N, K);

    auto cutlass_launch = [&](cudaStream_t stm) {
      launch_cutlass_gemm(cutlass_op, stm, M, N, K,
                          buf.dA, buf.dB, buf.dC_cutlass);
    };
    auto cublas_launch = [&](cudaStream_t stm) {
      CUBLAS_CHECK(cublasSetStream(cublas, stm));
      launch_cublas_gemm(cublas, M, N, K,
                         reinterpret_cast<__half const*>(buf.dA),
                         reinterpret_cast<__half const*>(buf.dB),
                         reinterpret_cast<__half*>(buf.dC_cublas));
    };

    // One-shot correctness check on the first shape (CUTLASS vs cuBLAS).
    if (!skip_validate) {
      cutlass_launch(stream);
      cublas_launch(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream));
      validate(buf, M, N);
      skip_validate = true;  // only validate once
    }

    double cublas_ms  = bench(cublas_launch,  stream, warmup, iters);
    double cutlass_ms = bench(cutlass_launch, stream, warmup, iters);

    double cublas_tf  = tflops(cublas_ms,  M, N, K);
    double cutlass_tf = tflops(cutlass_ms, M, N, K);
    double speedup    = cublas_ms / cutlass_ms;

    std::ostringstream label;
    label << "(" << M << "," << N << "," << K << "):";
    std::cout << std::left << std::setw(22) << label.str()
              << std::right << std::fixed << std::setprecision(3)
              << std::setw(12) << cublas_ms
              << std::setprecision(1) << std::setw(12) << cublas_tf
              << std::setprecision(3) << std::setw(13) << cutlass_ms
              << std::setprecision(1) << std::setw(13) << cutlass_tf
              << std::setprecision(2) << std::setw(9)  << speedup << "x"
              << std::defaultfloat << "\n";
  }

  CUBLAS_CHECK(cublasDestroy(cublas));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
