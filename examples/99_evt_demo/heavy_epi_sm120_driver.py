# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Driver for the hand-fused CuTeDSL heavy-epilogue kernel on sm_120 (RTX 5090).

Fused math (same as ``heavy_epi_torch_ext.cu``):

    D = tanh( SiLU(A @ B + Bias) * Scale + Aux )

All A/B/Bias/Scale/Aux/D are bf16; accumulator and epilogue are fp32.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.cute.testing as testing

from heavy_epi_sm120_kernel import Sm120HeavyEpiKernel


def heavy_epi_ref(A: torch.Tensor, B: torch.Tensor,
                  Bias: torch.Tensor, Scale: torch.Tensor,
                  Aux: torch.Tensor) -> torch.Tensor:
    """Torch-eager reference (bf16 compute, matches kernel semantics)."""
    x = torch.matmul(A, B).to(torch.float32)
    t0 = x + Bias.to(torch.float32)
    t1 = torch.nn.functional.silu(t0)
    t2 = t1 * Scale.to(torch.float32) + Aux.to(torch.float32)
    return torch.tanh(t2).to(torch.bfloat16)


def _make_cute(matrix_mnl_cpu: torch.Tensor, dtype):
    """Upload an (l, m, n)-shaped CPU tensor to GPU and wrap as a cute.Tensor."""
    cute_t, gpu_t = cutlass_torch.cute_tensor_like(
        matrix_mnl_cpu, dtype, is_dynamic_layout=True, assumed_align=16
    )
    return cute_t, gpu_t


def run_once(m: int, n: int, k: int,
             tile_shape_mnk=(128, 128, 64),
             warmup: int = 2, iters: int = 50,
             check: bool = True,
             tolerance: float = 2e-2):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required.")

    ab_dtype  = cutlass.BFloat16
    c_dtype   = cutlass.BFloat16
    acc_dtype = cutlass.Float32
    L = 1
    a_major = "k"      # A is (M, K), row-major K
    b_major = "k"      # B is (N, K), row-major K (i.e. N-major from the K*N view)
    c_major = "n"      # D/Bias/Scale/Aux are (M, N), row-major N

    # Host tensors.
    a_cpu    = cutlass_torch.matrix(L, m, k, a_major, ab_dtype)
    b_cpu    = cutlass_torch.matrix(L, n, k, b_major, ab_dtype)
    d_cpu    = cutlass_torch.matrix(L, m, n, c_major, c_dtype)
    bias_cpu = cutlass_torch.matrix(L, m, n, c_major, c_dtype)
    scl_cpu  = cutlass_torch.matrix(L, m, n, c_major, c_dtype)
    aux_cpu  = cutlass_torch.matrix(L, m, n, c_major, c_dtype)

    a_t,    a_gpu    = _make_cute(a_cpu,    ab_dtype)
    b_t,    b_gpu    = _make_cute(b_cpu,    ab_dtype)
    d_t,    d_gpu    = _make_cute(d_cpu,    c_dtype)
    bias_t, bias_gpu = _make_cute(bias_cpu, c_dtype)
    scl_t,  scl_gpu  = _make_cute(scl_cpu,  c_dtype)
    aux_t,  aux_gpu  = _make_cute(aux_cpu,  c_dtype)

    gemm = Sm120HeavyEpiKernel(acc_dtype=acc_dtype, tile_shape_mnk=tile_shape_mnk)
    max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
    stream = cutlass_torch.default_stream()

    t_jit0 = time.time()
    compiled = cute.compile(gemm, a_t, b_t, d_t, bias_t, scl_t, aux_t,
                            max_active_clusters, stream)
    print(f"[compile] {time.time() - t_jit0:.2f}s")

    if check:
        compiled(a_t, b_t, d_t, bias_t, scl_t, aux_t, stream)
        torch.cuda.synchronize()
        # cutlass_torch.matrix returns (M, K, L) / (N, K, L) / (M, N, L).
        A2  = a_gpu[..., 0]                       # (M, K)
        B2  = b_gpu[..., 0].t().contiguous()      # (K, N)
        Bi2 = bias_gpu[..., 0]                    # (M, N)
        Sc2 = scl_gpu[..., 0]
        Au2 = aux_gpu[..., 0]
        ref = heavy_epi_ref(A2, B2, Bi2, Sc2, Au2)
        got = d_gpu[..., 0]
        max_abs = (got.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
        print(f"[check] max|diff| = {max_abs:.4f}")
        assert max_abs <= tolerance, (
            f"Output mismatch: {max_abs} > {tolerance}")

    # Warmup
    for _ in range(warmup):
        compiled(a_t, b_t, d_t, bias_t, scl_t, aux_t, stream)
    torch.cuda.synchronize()

    # Time the kernel
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        compiled(a_t, b_t, d_t, bias_t, scl_t, aux_t, stream)
    t1.record()
    torch.cuda.synchronize()
    ms_per = t0.elapsed_time(t1) / iters
    gemm_flops = 2.0 * m * n * k
    tflops = gemm_flops / (ms_per * 1e-3) / 1e12
    print(f"[sm120-heavy-epi] M={m:<5d} N={n:<5d} K={k:<5d} "
          f"tile={tile_shape_mnk}  "
          f"{ms_per:.3f} ms/iter   {tflops:6.1f} TFLOPS (gemm-equivalent)")

    # torch eager baseline (unfused)
    A_ref     = a_gpu[..., 0]
    B_ref     = b_gpu[..., 0].t().contiguous()
    Bias_ref  = bias_gpu[..., 0]
    Scale_ref = scl_gpu[..., 0]
    Aux_ref   = aux_gpu[..., 0]

    for _ in range(warmup):
        _ = heavy_epi_ref(A_ref, B_ref, Bias_ref, Scale_ref, Aux_ref)
    torch.cuda.synchronize()
    t0.record()
    for _ in range(iters):
        _ = heavy_epi_ref(A_ref, B_ref, Bias_ref, Scale_ref, Aux_ref)
    t1.record()
    torch.cuda.synchronize()
    ms_eager = t0.elapsed_time(t1) / iters
    tflops_eager = gemm_flops / (ms_eager * 1e-3) / 1e12
    print(f"[torch-eager]     M={m:<5d} N={n:<5d} K={k:<5d} "
          f"{' ' * 19}{ms_eager:.3f} ms/iter   {tflops_eager:6.1f} TFLOPS")
    print(f"[speedup]         CuTeDSL / eager = {ms_eager / ms_per:.2f}x")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", type=str,
                   default="1024,1024,1024;2048,2048,2048;4096,4096,4096",
                   help="Semicolon-separated M,N,K triples")
    p.add_argument("--tile_shape_mnk", type=str, default="128,128,64")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--skip_check", action="store_true")
    args = p.parse_args()

    tile = tuple(int(x) for x in args.tile_shape_mnk.split(","))
    shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes.split(";")]
    torch.manual_seed(0)
    for (m, n, k) in shapes:
        print("=" * 78)
        run_once(m, n, k, tile_shape_mnk=tile,
                 warmup=args.warmup, iters=args.iters,
                 check=(not args.skip_check))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
