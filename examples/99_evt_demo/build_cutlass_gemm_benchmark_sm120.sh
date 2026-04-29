#!/bin/bash
# Build the CUTLASS fp16 GEMM vs cuBLAS benchmark for RTX 5090 (sm_120).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUTLASS_ROOT="$(cd "${HERE}/../.." && pwd)"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"

"${NVCC}" -std=c++17 -O3                                  \
  -gencode arch=compute_120,code=sm_120                   \
  --expt-relaxed-constexpr                                \
  -I"${CUTLASS_ROOT}/include"                             \
  -I"${CUTLASS_ROOT}/tools/util/include"                  \
  "${HERE}/cutlass_gemm_benchmark_sm120.cu"               \
  -o "${HERE}/cutlass_gemm_benchmark_sm120"               \
  -lcublas

echo "Built: ${HERE}/cutlass_gemm_benchmark_sm120"
