"""Benchmark: CUTLASS fp16 GEMM (via PyTorch C++ extension) vs torch.mm.

Loads ``cutlass_gemm_ext.cu`` with ``torch.utils.cpp_extension.load`` on first
run (JIT compile + cache under ``~/.cache/torch_extensions/``), then exposes
``cutlass_gemm_ext.gemm_fp16(A, B)`` / ``gemm_fp16_out(A, B, C)`` which compute
``A @ B.T`` on the current CUDA stream.

Both the CUTLASS and torch.mm launches are scheduled on the same stream so
``torch.cuda.Event`` timings are apples-to-apples.

Usage
-----
    python examples/99_evt_demo/cutlass_gemm_pybind_benchmark.py
    python examples/99_evt_demo/cutlass_gemm_pybind_benchmark.py \
        --shapes "4096,4096,4096;8192,8192,8192" --iters 50
"""

import argparse
import math
import os
import sys
from typing import List, Tuple

import torch
from torch.utils.cpp_extension import load


HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

DEFAULT_SHAPES: List[Tuple[int, int, int]] = [
    (4096, 4096, 4096),
    (4096, 8192, 4096),
    (8192, 8192, 4096),
    (2048, 8192, 8192),
    (16384, 4096, 4096),
    (1024, 14336, 4096),
    (4096, 14336, 4096),
]


def build_extension(verbose: bool = False, arch: str = "120"):
    """JIT build the CUTLASS extension. Cached across runs."""
    extra_cuda_cflags = [
        "-std=c++17",
        "-O3",
        "--expt-relaxed-constexpr",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
    ]
    ext = load(
        name="cutlass_gemm_ext",
        sources=[os.path.join(HERE, "cutlass_gemm_ext.cu")],
        extra_include_paths=[
            os.path.join(CUTLASS_ROOT, "include"),
            os.path.join(CUTLASS_ROOT, "tools", "util", "include"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=verbose,
    )
    return ext


def parse_shape_list(s: str) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        parts = [int(x) for x in tok.split(",")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"Shape '{tok}' must be M,N,K (comma-separated)."
            )
        out.append((parts[0], parts[1], parts[2]))
    return out


def tflops(ms: float, M: int, N: int, K: int) -> float:
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12


def bench(fn, stream, warmup: int, iters: int) -> float:
    """Time ``fn`` on ``stream`` using CUDA events, returns avg ms."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record(stream)
    for _ in range(iters):
        fn()
    e.record(stream)
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", type=str, default="")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--arch", type=str, default="120",
                    help="SM arch number (default 120 for RTX 5090)")
    ap.add_argument("--verbose-build", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    dev = torch.device("cuda:0")
    major, minor = torch.cuda.get_device_capability(dev)
    print(f"Device: {torch.cuda.get_device_name(dev)} (sm_{major}{minor})",
          flush=True)

    print(f"Building CUTLASS extension (arch=sm_{args.arch}) ...", flush=True)
    ext = build_extension(verbose=args.verbose_build, arch=args.arch)
    print("Built.", flush=True)

    shapes = parse_shape_list(args.shapes) if args.shapes else DEFAULT_SHAPES

    stream = torch.cuda.current_stream(dev)

    # One-shot correctness check on the first shape.
    if not args.skip_validate:
        M0, N0, K0 = shapes[0]
        A0 = torch.randn((M0, K0), dtype=torch.float16, device=dev)
        B0 = torch.randn((N0, K0), dtype=torch.float16, device=dev)
        C_cutlass = ext.gemm_fp16(A0, B0)
        C_torch   = torch.mm(A0, B0.t())
        diff = (C_cutlass.float() - C_torch.float()).abs().max().item()
        rel = diff / max(1e-6, C_torch.float().abs().mean().item())
        print(f"Correctness on ({M0},{N0},{K0}): "
              f"max|diff|={diff:.4f}, rel={rel:.2e}", flush=True)

    header = (
        f"{'shape (M,N,K)':<22} "
        f"{'torch.mm ms':>12} {'torch TF':>10} "
        f"{'CUTLASS ms':>12} {'CUTLASS TF':>11} "
        f"{'speedup':>9}"
    )
    print()
    print(header)
    print("-" * len(header))

    for M, N, K in shapes:
        A = torch.randn((M, K), dtype=torch.float16, device=dev)
        B = torch.randn((N, K), dtype=torch.float16, device=dev)
        C = torch.empty((M, N), dtype=torch.float16, device=dev)

        def cutlass_fn():
            ext.gemm_fp16_out(A, B, C)

        def torch_fn():
            torch.mm(A, B.t(), out=C)

        torch_ms   = bench(torch_fn,   stream, args.warmup, args.iters)
        cutlass_ms = bench(cutlass_fn, stream, args.warmup, args.iters)

        torch_tf   = tflops(torch_ms,   M, N, K)
        cutlass_tf = tflops(cutlass_ms, M, N, K)
        speedup    = torch_ms / cutlass_ms

        print(
            f"({M},{N},{K}):".ljust(22)
            + f" {torch_ms:>12.3f}"
            + f" {torch_tf:>10.1f}"
            + f" {cutlass_ms:>12.3f}"
            + f" {cutlass_tf:>11.1f}"
            + f" {speedup:>8.2f}x",
            flush=True,
        )


if __name__ == "__main__":
    main()
