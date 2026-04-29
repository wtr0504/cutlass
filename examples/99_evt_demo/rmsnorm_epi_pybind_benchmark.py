"""Benchmark CUTLASS fused matmul + RMSNorm vs torch eager / torch.compile.

Computation:

    D = rmsnorm(A @ B.T) * gamma

    A     : (M, K) bf16
    B     : (N, K) bf16   (torch.nn.Linear weight convention)
    gamma : (N,)   bf16   (per-feature RMSNorm scale)
    D     : (M, N) bf16

where, in pure PyTorch:

    def rmsnorm(x, gamma, eps=1e-6):
        rms_inv = (x.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
        return x * rms_inv * gamma

CUTLASS implementation: two kernels.
  Kernel 1 — EVT GEMM that writes Y AND atomically accumulates per-row
             sum-of-squares into a small (M,) fp32 buffer (the EVT tree
             tees off Square→VisitorRowReduction as a side branch from the
             accumulator while the main branch stores Y unchanged).
  Kernel 2 — Custom finalize+normalize: rms_inv[m] = rsqrt(sum_sq[m]/N+eps),
             then writes D[m,n] = Y[m,n] * rms_inv[m] * gamma[n].

vs torch.compile, this saves the entire variance-computation pass over Y
(sum-of-squares is computed for free in the GEMM epilogue's register tile)
and one kernel launch.

Comparison paths
----------------
  eager   : torch.nn.functional.linear(A, B) → pow → mean → rsqrt → mul → mul.
  compile : torch.compile(eager_ref, mode="max-autotune"). Inductor typically
            produces 1 cuBLAS matmul + 1-2 Triton kernels for variance/normalize.
  cutlass : EVT GEMM with side-effect row-reduction + custom finalize kernel.

Usage
-----
    python examples/99_evt_demo/rmsnorm_epi_pybind_benchmark.py
    python examples/99_evt_demo/rmsnorm_epi_pybind_benchmark.py \\
        --shapes "2048,4096,4096;4096,8192,4096" --iters 50
    python examples/99_evt_demo/rmsnorm_epi_pybind_benchmark.py --no-compile
"""

import argparse
import os
from typing import Callable, List, Tuple

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

DEFAULT_SHAPES: List[Tuple[int, int, int]] = [
    (4096, 4096, 4096),
    (4096, 8192, 4096),
    (8192, 8192, 4096),
    (2048, 8192, 8192),
    (1024, 14336, 4096),
    (4096, 14336, 4096),
    (7697, 27304, 5120),
]

EPS = 1e-6


# ---------------------------------------------------------------------------
# Extension builder
# ---------------------------------------------------------------------------

def build_extension(verbose: bool = False, arch: str = "120"):
    extra_cuda_cflags = [
        "-std=c++17",
        "-O3",
        "--expt-relaxed-constexpr",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
    ]
    return load(
        name="rmsnorm_epi_torch_ext",
        sources=[os.path.join(HERE, "rmsnorm_epi_torch_ext.cu")],
        extra_include_paths=[
            os.path.join(CUTLASS_ROOT, "include"),
            os.path.join(CUTLASS_ROOT, "tools", "util", "include"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Reference implementation (matches CUTLASS computation exactly)
# ---------------------------------------------------------------------------

def rmsnorm_ref(A: torch.Tensor,
                B: torch.Tensor,
                gamma: torch.Tensor,
                eps: float = EPS) -> torch.Tensor:
    """Pure PyTorch reference: D = rmsnorm(A @ B.T) * gamma."""
    Y = torch.nn.functional.linear(A, B).float()                     # (M, N)
    rms_inv = (Y.pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()    # (M, 1)
    return (Y * rms_inv * gamma.float()).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_shape_list(s: str) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        parts = [int(x) for x in tok.split(",")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"Shape '{tok}' must be M,N,K (comma-separated).")
        out.append((parts[0], parts[1], parts[2]))
    return out


def bench(fn: Callable[[], None], warmup: int, iters: int) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def gemm_tflops(ms: float, M: int, N: int, K: int) -> float:
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", type=str, default="")
    ap.add_argument("--iters",  type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--arch",   type=str, default="120")
    ap.add_argument("--verbose-build", action="store_true")
    ap.add_argument("--skip-validate", action="store_true",
                    help="Skip per-shape CUTLASS correctness validation")
    ap.add_argument("--rel-tol", type=float, default=1e-2,
                    help="Mean-relative-error tolerance for validation pass "
                         "(default 1e-2, suited to bf16)")
    ap.add_argument("--no-compile", action="store_true",
                    help="Skip torch.compile benchmarking")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    dev = torch.device("cuda:0")
    major, minor = torch.cuda.get_device_capability(dev)
    print(f"Device: {torch.cuda.get_device_name(dev)} (sm_{major}{minor})",
          flush=True)
    print(f"torch:  {torch.__version__}", flush=True)

    print(f"Building CUTLASS rmsnorm extension (arch=sm_{args.arch}) ...",
          flush=True)
    ext = build_extension(verbose=args.verbose_build, arch=args.arch)
    print("  built.", flush=True)

    shapes = parse_shape_list(args.shapes) if args.shapes else DEFAULT_SHAPES

    compiled_fn = None
    if not args.no_compile:
        print("torch.compile(rmsnorm_ref) ... (first call will JIT)", flush=True)
        compiled_fn = torch.compile(rmsnorm_ref, mode="max-autotune",
                                    dynamic=False)

    # ------------------------------------------------------------------
    # Per-shape CUTLASS correctness validation against the PyTorch reference.
    # Deterministic seed per shape so failures reproduce.
    # ------------------------------------------------------------------
    if not args.skip_validate:
        REL_TOL = args.rel_tol
        print(f"\n[CUTLASS correctness vs PyTorch reference, rel-tol={REL_TOL:.0e}]",
              flush=True)
        v_header = (
            f"{'shape (M,N,K)':<22}"
            f" {'max|diff|':>10} {'mean|diff|':>11} {'rel':>10}  verdict"
        )
        print(v_header)
        print("-" * len(v_header))

        all_pass = True
        for idx, (M, N, K) in enumerate(shapes):
            torch.manual_seed(0xC07A55 + idx)
            A     = torch.randn((M, K), dtype=torch.bfloat16, device=dev)
            B     = torch.randn((N, K), dtype=torch.bfloat16, device=dev)
            gamma = torch.randn((N,),   dtype=torch.bfloat16, device=dev)
            D_cut = torch.empty((M, N), dtype=torch.bfloat16, device=dev)

            ext.rmsnorm_matmul_out(A, B, gamma, D_cut, EPS)
            D_ref = rmsnorm_ref(A, B, gamma, EPS)

            diff = (D_cut.float() - D_ref.float()).abs()
            ref_mean_abs = D_ref.float().abs().mean().item()
            rel = (diff.mean() / max(1e-6, ref_mean_abs)).item()
            ok  = rel <= REL_TOL
            all_pass = all_pass and ok
            print(
                f"({M},{N},{K}):".ljust(22)
                + f" {diff.max().item():>10.4f}"
                + f" {diff.mean().item():>11.5f}"
                + f" {rel:>10.2e}"
                + ("  PASS" if ok else "  FAIL"),
                flush=True,
            )

            if idx == 0 and compiled_fn is not None:
                D_cmp = compiled_fn(A, B, gamma)
                diff2 = (D_cmp.float() - D_ref.float()).abs()
                print(f"  (torch.compile vs eager: "
                      f"max|diff|={diff2.max().item():.4f})",
                      flush=True)

            del A, B, gamma, D_cut, D_ref, diff
        print(f"  -> {'all shapes PASS' if all_pass else 'SOME SHAPES FAILED'}",
              flush=True)
        if not all_pass:
            raise RuntimeError("CUTLASS correctness check failed on at least "
                               "one shape (see table above).")

    # ------------------------------------------------------------------
    # Benchmark table
    # ------------------------------------------------------------------
    header = (
        f"{'shape (M,N,K)':<22}"
        f" {'eager ms':>10} {'eager TF':>10}"
        f" {'compile ms':>11} {'compile TF':>11}"
        f" {'CUTLASS ms':>11} {'CUTLASS TF':>11}"
        f" {'vs eager':>9} {'vs compile':>11}"
    )
    print(f"\n[matmul+RMSNorm fused — bf16, eps={EPS}]")
    print(header)
    print("-" * len(header))

    for M, N, K in shapes:
        A     = torch.randn((M, K), dtype=torch.bfloat16, device=dev)
        B     = torch.randn((N, K), dtype=torch.bfloat16, device=dev)
        gamma = torch.randn((N,),   dtype=torch.bfloat16, device=dev)
        D     = torch.empty((M, N), dtype=torch.bfloat16, device=dev)

        def cutlass_fn():
            ext.rmsnorm_matmul_out(A, B, gamma, D, EPS)

        def eager_fn():
            rmsnorm_ref(A, B, gamma)

        cutlass_ms = bench(cutlass_fn, args.warmup, args.iters)
        eager_ms   = bench(eager_fn,   args.warmup, args.iters)

        if compiled_fn is not None:
            def compile_fn():
                compiled_fn(A, B, gamma)
            compile_ms = bench(compile_fn, args.warmup, args.iters)
        else:
            compile_ms = float("nan")

        cutlass_tf = gemm_tflops(cutlass_ms, M, N, K)
        eager_tf   = gemm_tflops(eager_ms,   M, N, K)
        compile_tf = (gemm_tflops(compile_ms, M, N, K)
                      if compile_ms == compile_ms else float("nan"))

        speedup_e = eager_ms   / cutlass_ms
        speedup_c = (compile_ms / cutlass_ms
                     if compile_ms == compile_ms else float("nan"))

        compile_ms_str = (f"{compile_ms:>11.3f}" if compile_ms == compile_ms
                          else f"{'n/a':>11}")
        compile_tf_str = (f"{compile_tf:>11.1f}" if compile_tf == compile_tf
                          else f"{'n/a':>11}")
        speedup_c_str  = (f"{speedup_c:>10.2f}x" if speedup_c == speedup_c
                          else f"{'n/a':>11}")

        print(
            f"({M},{N},{K}):".ljust(22)
            + f" {eager_ms:>10.3f} {eager_tf:>10.1f}"
            + f" {compile_ms_str} {compile_tf_str}"
            + f" {cutlass_ms:>11.3f} {cutlass_tf:>11.1f}"
            + f" {speedup_e:>8.2f}x"
            + f" {speedup_c_str}",
            flush=True,
        )


if __name__ == "__main__":
    main()
