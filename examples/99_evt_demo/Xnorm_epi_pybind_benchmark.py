"""Benchmark CUTLASS fused matmul + {RMSNorm, LayerNorm, L2Norm} vs torch.

Computation:

    D = norm(A @ B.T) * gamma [+ beta]

    A     : (M, K) bf16
    B     : (N, K) bf16   (torch.nn.Linear weight convention)
    gamma : (N,)   bf16   (per-feature affine scale, optional)
    beta  : (N,)   bf16   (per-feature affine bias, LayerNorm only)
    D     : (M, N) bf16

Norm types (selected at launch via the `norm_type` argument):

    rmsnorm   : D = Y * rsqrt(mean(Y^2) + eps) * gamma
    layernorm : D = (Y - mean(Y)) * rsqrt(var(Y) + eps) * gamma + beta
    l2norm    : D = Y * rsqrt(sum(Y^2) + eps) * gamma            (no /N)

CUTLASS implementation: two kernels.
  Kernel 1 — EVT GEMM that writes Y AND atomically accumulates per-row sum
             AND per-row sum-of-squares into two small (M,) fp32 buffers
             (the EVT tree tees off Square→ColReduce as a side branch from
             the accumulator, then chains a second ColReduce on the raw Accum,
             while the main branch stores Y unchanged).
  Kernel 2 — Templated finalize: dispatched on norm_type. Computes inv_std
             (and mean for layernorm) per row, then writes D.

vs torch.compile, this saves the entire variance pass over Y (mean and
sum-of-squares are computed for free in the GEMM epilogue's register tile)
and one kernel launch — for ALL three norm types from the same EVT GEMM.

Usage
-----
    python examples/99_evt_demo/Xnorm_epi_pybind_benchmark.py
    python examples/99_evt_demo/Xnorm_epi_pybind_benchmark.py \\
        --norms rmsnorm,layernorm --shapes "2048,4096,4096"
    python examples/99_evt_demo/Xnorm_epi_pybind_benchmark.py --no-compile
"""

import argparse
import os
from typing import Callable, List, Optional, Tuple

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
    (1924, 27304, 5120),
    (7696, 27304, 5120),
]

DEFAULT_NORMS = ("rmsnorm", "layernorm", "l2norm")
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
        name="xnorm_epi_torch_ext",
        sources=[os.path.join(HERE, "Xnorm_epi_torch_ext.cu")],
        extra_include_paths=[
            os.path.join(CUTLASS_ROOT, "include"),
            os.path.join(CUTLASS_ROOT, "tools", "util", "include"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Reference implementations (match CUTLASS computation exactly)
# ---------------------------------------------------------------------------

def _matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.linear(A, B).float()           # (M, N)


def rmsnorm_ref(A, B, gamma, beta=None, eps: float = EPS) -> torch.Tensor:
    Y = _matmul(A, B)
    inv_std = (Y.pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()
    out = Y * inv_std
    if gamma is not None:
        out = out * gamma.float()
    return out.to(torch.bfloat16)


def layernorm_ref(A, B, gamma, beta=None, eps: float = EPS) -> torch.Tensor:
    Y = _matmul(A, B)
    mean = Y.mean(dim=-1, keepdim=True)
    var = Y.var(dim=-1, keepdim=True, unbiased=False)
    inv_std = (var + eps).rsqrt()
    out = (Y - mean) * inv_std
    if gamma is not None:
        out = out * gamma.float()
    if beta is not None:
        out = out + beta.float()
    return out.to(torch.bfloat16)


def l2norm_ref(A, B, gamma, beta=None, eps: float = EPS) -> torch.Tensor:
    Y = _matmul(A, B)
    inv_std = (Y.pow(2).sum(dim=-1, keepdim=True) + eps).rsqrt()
    out = Y * inv_std
    if gamma is not None:
        out = out * gamma.float()
    return out.to(torch.bfloat16)


REFS = {
    "rmsnorm":   rmsnorm_ref,
    "layernorm": layernorm_ref,
    "l2norm":    l2norm_ref,
}


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


def parse_norm_list(s: str) -> List[str]:
    norms = [t.strip().lower() for t in s.split(",") if t.strip()]
    for n in norms:
        if n not in REFS:
            raise argparse.ArgumentTypeError(
                f"Unknown norm '{n}'. Choices: {list(REFS)}")
    return norms


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


def make_inputs(M: int, N: int, K: int, norm: str, dev, seed: int):
    torch.manual_seed(seed)
    A     = torch.randn((M, K), dtype=torch.bfloat16, device=dev)
    B     = torch.randn((N, K), dtype=torch.bfloat16, device=dev)
    gamma = torch.randn((N,),   dtype=torch.bfloat16, device=dev)
    beta  = (torch.randn((N,), dtype=torch.bfloat16, device=dev)
             if norm == "layernorm" else None)
    D     = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    return A, B, gamma, beta, D


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", type=str, default="")
    ap.add_argument("--norms",  type=str, default=",".join(DEFAULT_NORMS),
                    help="Comma-separated list: rmsnorm,layernorm,l2norm")
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

    print(f"Building CUTLASS Xnorm extension (arch=sm_{args.arch}) ...",
          flush=True)
    ext = build_extension(verbose=args.verbose_build, arch=args.arch)
    print("  built.", flush=True)

    shapes = parse_shape_list(args.shapes) if args.shapes else DEFAULT_SHAPES
    norms  = parse_norm_list(args.norms)

    # Pre-compile reference paths if requested.
    compiled_refs: dict = {}
    if not args.no_compile:
        print("torch.compile(<ref>) for each norm ... (first call will JIT)",
              flush=True)
        for n in norms:
            compiled_refs[n] = torch.compile(REFS[n], mode="max-autotune",
                                             dynamic=False)

    # ------------------------------------------------------------------
    # Per-(norm,shape) CUTLASS correctness validation.
    # ------------------------------------------------------------------
    if not args.skip_validate:
        REL_TOL = args.rel_tol
        print(f"\n[CUTLASS correctness vs PyTorch reference, rel-tol={REL_TOL:.0e}]",
              flush=True)
        v_header = (
            f"{'norm':<10} {'shape (M,N,K)':<22}"
            f" {'max|diff|':>10} {'mean|diff|':>11} {'rel':>10}  verdict"
        )
        print(v_header)
        print("-" * len(v_header))

        all_pass = True
        for norm in norms:
            for idx, (M, N, K) in enumerate(shapes):
                A, B, gamma, beta, D_cut = make_inputs(
                    M, N, K, norm, dev, seed=0xC07A55 + idx)

                ext.xnorm_matmul_out(A, B, gamma, beta, D_cut, norm, EPS)
                D_ref = REFS[norm](A, B, gamma, beta, EPS)

                diff = (D_cut.float() - D_ref.float()).abs()
                ref_mean_abs = D_ref.float().abs().mean().item()
                rel = (diff.mean() / max(1e-6, ref_mean_abs)).item()
                ok  = rel <= REL_TOL
                all_pass = all_pass and ok
                print(
                    f"{norm:<10} ({M},{N},{K}):".ljust(33)
                    + f" {diff.max().item():>10.4f}"
                    + f" {diff.mean().item():>11.5f}"
                    + f" {rel:>10.2e}"
                    + ("  PASS" if ok else "  FAIL"),
                    flush=True,
                )

                del A, B, gamma, beta, D_cut, D_ref, diff
        print(f"  -> {'all PASS' if all_pass else 'SOME FAILED'}", flush=True)
        if not all_pass:
            raise RuntimeError(
                "CUTLASS correctness check failed (see table above).")

    # ------------------------------------------------------------------
    # Benchmark table — one block per norm type.
    # ------------------------------------------------------------------
    for norm in norms:
        ref_fn  = REFS[norm]
        comp_fn = compiled_refs.get(norm)

        header = (
            f"{'shape (M,N,K)':<22}"
            f" {'eager ms':>10} {'eager TF':>10}"
            f" {'compile ms':>11} {'compile TF':>11}"
            f" {'CUTLASS ms':>11} {'CUTLASS TF':>11}"
            f" {'vs eager':>9} {'vs compile':>11}"
        )
        print(f"\n[matmul+{norm} fused — bf16, eps={EPS}]")
        print(header)
        print("-" * len(header))

        for M, N, K in shapes:
            A, B, gamma, beta, D = make_inputs(M, N, K, norm, dev, seed=0)

            def cutlass_fn():
                ext.xnorm_matmul_out(A, B, gamma, beta, D, norm, EPS)

            def eager_fn():
                ref_fn(A, B, gamma, beta)

            cutlass_ms = bench(cutlass_fn, args.warmup, args.iters)
            eager_ms   = bench(eager_fn,   args.warmup, args.iters)

            if comp_fn is not None:
                def compile_fn():
                    comp_fn(A, B, gamma, beta)
                compile_ms = bench(compile_fn, args.warmup, args.iters)
            else:
                compile_ms = float("nan")

            cutlass_tf = gemm_tflops(cutlass_ms, M, N, K)
            eager_tf   = gemm_tflops(eager_ms,   M, N, K)
            compile_tf = (gemm_tflops(compile_ms, M, N, K)
                          if compile_ms == compile_ms else float("nan"))

            speedup_e = eager_ms / cutlass_ms
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
