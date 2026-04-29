"""Benchmark: CuTeDSL Sm120 dense_gemm vs torch.mm (cuBLAS) on RTX 5090.

Drives the Blackwell-Geforce (SM120) persistent GEMM implemented in
examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py across a handful of
transformer-sized shapes and compares against torch.mm, which dispatches to
cuBLAS on consumer Blackwell.

Why this differs from the Hopper benchmark
------------------------------------------
The Hopper kernel (``HopperWgmmaGemmPersistentKernel``) uses a 2D CTA tile
(M, N) with K=64 implicit and supports cluster-shape tuning. The SM120
``Sm120GemmKernel`` uses a *3D* CTA tile ``(M, N, K)`` and is hard-coded to a
``(1, 1, 1)`` cluster, so we autotune over tile_shape_mnk only.

Allowed tile shapes on SM120 (from the example's argparse ``choices``):
    (64, 64,  64), (64, 128, 64), (128, 64,  64),
    (128, 128, 64), (128, 256, 64), (128, 128, 128)

Layout
------
We pass A as (M, K) row-major and B as (N, K) row-major and compute
    C = A @ B.T
which matches the CuTeDSL kernel's default a_major="k", b_major="k",
c_major="n" and is exactly ``torch.mm(A, B.t())``.

Timing
------
Both the CuTeDSL kernel and torch.mm are scheduled on the same CUDA stream
(the current torch stream) so that torch.cuda.Event based timers capture
them correctly. For each tile candidate we compile once on a reference
shape and re-use the compiled artifact across shapes, which is safe because
the example kernel is compiled with is_dynamic_layout=True.

Usage
-----
    python examples/99_evt_demo/cutedsl_gemm_benchmark_sm120.py
    python examples/99_evt_demo/cutedsl_gemm_benchmark_sm120.py --quick
    python examples/99_evt_demo/cutedsl_gemm_benchmark_sm120.py \
        --shapes "4096,4096,4096;8192,8192,8192"
"""

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch


HERE = os.path.dirname(os.path.abspath(__file__))
# examples/99_evt_demo/ -> examples/python/CuTeDSL/blackwell_geforce
BG_DIR = os.path.abspath(
    os.path.join(HERE, "..", "python", "CuTeDSL", "blackwell_geforce")
)
if BG_DIR not in sys.path:
    sys.path.insert(0, BG_DIR)

from dense_gemm import Sm120GemmKernel  # noqa: E402


DEFAULT_SHAPES: List[Tuple[int, int, int]] = [
    (4096, 4096, 4096),
    (4096, 8192, 4096),
    (8192, 8192, 4096),
    (2048, 8192, 8192),
    (16384, 4096, 4096),
    (1024, 14336, 4096),
    (4096, 14336, 4096),
]


# Tile shapes allowed by Sm120GemmKernel (see argparse choices in the example).
ALL_TILES: List[Tuple[int, int, int]] = [
    (128, 256, 64),
    (128, 128, 64),
    (128, 128, 128),
    (128, 64, 64),
    (64, 128, 64),
    (64, 64, 64),
]

QUICK_TILES: List[Tuple[int, int, int]] = [
    (128, 256, 64),
    (128, 128, 64),
    (128, 128, 128),
]


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


def alloc_shape(
    M: int, N: int, K: int, cutlass_dtype
) -> Tuple[object, object, object, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create (a_cute, b_cute, c_cute, a_gpu2d, b_gpu2d, c_gpu2d) for one shape.

    The 2D GPU views are (M,K), (N,K), (M,N) row-major fp16 tensors that share
    storage with the cute tensors. The cute tensors carry an L=1 batch axis.
    """
    a_cpu = cutlass_torch.matrix(1, M, K, False, cutlass_dtype)  # k-major
    b_cpu = cutlass_torch.matrix(1, N, K, False, cutlass_dtype)  # k-major
    c_cpu = cutlass_torch.matrix(1, M, N, False, cutlass_dtype)  # n-major
    a_cute, a_gpu = cutlass_torch.cute_tensor_like(
        a_cpu, cutlass_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_cute, b_gpu = cutlass_torch.cute_tensor_like(
        b_cpu, cutlass_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_cute, c_gpu = cutlass_torch.cute_tensor_like(
        c_cpu, cutlass_dtype, is_dynamic_layout=True, assumed_align=16
    )
    # Strip the trailing L=1 axis so the views are 2D for torch.mm.
    return a_cute, b_cute, c_cute, a_gpu[..., 0], b_gpu[..., 0], c_gpu[..., 0]


def tile_fits_shape(tile: Tuple[int, int, int], M: int, N: int, K: int) -> bool:
    tM, tN, tK = tile
    return (M % tM == 0) and (N % tN == 0) and (K % tK == 0)


def tflops(ms: float, M: int, N: int, K: int) -> float:
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12


def bench_stream(fn, stream, warmup=5, iters=20) -> float:
    """Bench ``fn`` on ``stream`` using CUDA events. Returns avg ms."""
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


def compile_all(
    tiles: List[Tuple[int, int, int]],
    cutlass_dtype,
    stream_obj,
    verbose: bool,
) -> Dict[Tuple[int, int, int], object]:
    """Compile every tile on a reference shape and return {tile: compiled}.

    SM120 has a fixed (1,1,1) cluster so we only autotune over tile_shape_mnk.
    """
    ref_M, ref_N, ref_K = 4096, 4096, 4096
    ref_a, ref_b, ref_c, *_ = alloc_shape(ref_M, ref_N, ref_K, cutlass_dtype)

    hw = cutlass.utils.HardwareInfo()
    # Sm120GemmKernel always uses cluster (1,1,1), so cluster_size = 1.
    max_active_clusters = hw.get_max_active_clusters(1)

    compiled: Dict[Tuple[int, int, int], object] = {}
    for tile in tiles:
        if verbose:
            print(f"  [compile] tile={tile} ...", end="", flush=True)
        try:
            gemm = Sm120GemmKernel(cutlass.Float32, tile)
            compiled[tile] = cute.compile(
                gemm, ref_a, ref_b, ref_c, max_active_clusters, stream_obj
            )
            if verbose:
                print(" ok", flush=True)
        except Exception as ex:  # noqa: BLE001
            if verbose:
                print(f" skip ({type(ex).__name__}: {ex})", flush=True)
    return compiled


def validate_once(compiled, a_cute, b_cute, c_cute, a_gpu, b_gpu, c_gpu, stream_obj):
    """Run one config once and compare against torch.mm."""
    compiled(a_cute, b_cute, c_cute, stream_obj)
    torch.cuda.synchronize()
    ref = torch.mm(a_gpu.float(), b_gpu.float().t()).to(c_gpu.dtype)
    diff = (c_gpu.float() - ref.float()).abs().max().item()
    rel = diff / max(1e-6, ref.float().abs().mean().item())
    return diff, rel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--shapes", type=str, default="",
        help="Custom shapes as 'M,N,K;M,N,K;...' (default: built-in list)",
    )
    ap.add_argument("--iters", type=int, default=20, help="timed iterations")
    ap.add_argument("--warmup", type=int, default=5, help="warmup iterations")
    ap.add_argument(
        "--quick", action="store_true",
        help="Use a small tile subset to cut compile time.",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--skip-validate", action="store_true",
        help="Skip the one-shot correctness check on the first shape.",
    )
    ap.add_argument(
        "--dtype", choices=["fp16", "bf16"], default="fp16",
        help="Input/output dtype (accumulator is always fp32).",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    # Quick sanity check for compute capability.
    major, minor = torch.cuda.get_device_capability(0)
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name} (sm_{major}{minor})", flush=True)
    if (major, minor) != (12, 0):
        print(
            f"  WARNING: Sm120GemmKernel targets sm_120 (e.g. RTX 5090). "
            f"Current device is sm_{major}{minor}.",
            flush=True,
        )

    shapes = parse_shape_list(args.shapes) if args.shapes else DEFAULT_SHAPES
    tiles = QUICK_TILES if args.quick else ALL_TILES
    cutlass_dtype = cutlass.Float16 if args.dtype == "fp16" else cutlass.BFloat16

    # Use the default torch stream so that CUDA events record both the cute
    # kernel and torch.mm launches correctly.
    torch_stream = torch.cuda.current_stream()
    stream_obj = cuda.CUstream(torch_stream.cuda_stream)

    print(
        f"Compiling CuTeDSL SM120 candidates ({len(tiles)} tiles) ...",
        flush=True,
    )
    compiled = compile_all(tiles, cutlass_dtype, stream_obj, args.verbose)
    print(f"Compiled {len(compiled)} / {len(tiles)} candidates.", flush=True)
    if not compiled:
        raise RuntimeError("No CuTeDSL candidate compiled successfully.")

    # One-shot correctness check using the first candidate & first shape.
    if not args.skip_validate:
        # Pick a candidate that divides the first shape cleanly if possible.
        M0, N0, K0 = shapes[0]
        val_tile = next(
            (t for t in compiled if tile_fits_shape(t, M0, N0, K0)),
            next(iter(compiled)),
        )
        # If chosen tile doesn't divide cleanly, fall back to a 4096-cube.
        if not tile_fits_shape(val_tile, M0, N0, K0):
            M0 = N0 = K0 = 4096
        a_c, b_c, c_c, a_g, b_g, c_g = alloc_shape(M0, N0, K0, cutlass_dtype)
        diff, rel = validate_once(
            compiled[val_tile], a_c, b_c, c_c, a_g, b_g, c_g, stream_obj
        )
        print(
            f"Correctness on ({M0},{N0},{K0}) via tile={val_tile}: "
            f"max|diff|={diff:.4f}, rel={rel:.2e}",
            flush=True,
        )

    header = (
        f"{'shape (M,N,K)':<22} "
        f"{'cuBLAS ms':>10} {'cuBLAS TF':>10} "
        f"{'CuTeDSL ms':>11} {'CuTeDSL TF':>11} "
        f"{'speedup':>9}  {'best tile'}"
    )
    print()
    print(header)
    print("-" * len(header))

    for M, N, K in shapes:
        a_c, b_c, c_c, a_g, b_g, c_g = alloc_shape(M, N, K, cutlass_dtype)

        # Autotune over compiled candidates on this shape; only pick tiles
        # that divide the shape cleanly since the example disallows OOB
        # tiles.
        best_tile: Optional[Tuple[int, int, int]] = None
        best_compiled = None
        best_ms = math.inf
        for tile, cpl in compiled.items():
            if not tile_fits_shape(tile, M, N, K):
                if args.verbose:
                    print(f"  [{tile}] skip: does not divide ({M},{N},{K})", flush=True)
                continue
            try:
                def run():
                    cpl(a_c, b_c, c_c, stream_obj)
                ms = bench_stream(run, torch_stream, warmup=2, iters=5)
            except Exception as ex:  # noqa: BLE001
                if args.verbose:
                    print(f"  [{tile}] run-fail {type(ex).__name__}: {ex}", flush=True)
                continue
            if args.verbose:
                print(f"  [{tile}] {ms:.3f} ms", flush=True)
            if ms < best_ms:
                best_ms = ms
                best_tile = tile
                best_compiled = cpl

        if best_compiled is None:
            print(f"({M},{N},{K}): all candidates failed", flush=True)
            continue

        def cute_fn():
            best_compiled(a_c, b_c, c_c, stream_obj)
        cute_ms = bench_stream(cute_fn, torch_stream, warmup=args.warmup, iters=args.iters)

        # torch.mm baseline writes into the same buffer; we only care about
        # raw throughput, not the epilogue details.
        def torch_fn():
            torch.mm(a_g, b_g.t(), out=c_g)
        torch_ms = bench_stream(torch_fn, torch_stream, warmup=args.warmup, iters=args.iters)

        cfg_str = f"T{best_tile[0]}x{best_tile[1]}x{best_tile[2]}"
        print(
            f"({M},{N},{K}):".ljust(22)
            + f" {torch_ms:>10.3f}"
            + f" {tflops(torch_ms, M, N, K):>10.1f}"
            + f" {cute_ms:>11.3f}"
            + f" {tflops(cute_ms, M, N, K):>11.1f}"
            + f" {torch_ms / cute_ms:>8.2f}x"
            + f"  {cfg_str}",
            flush=True,
        )


if __name__ == "__main__":
    main()
