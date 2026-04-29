"""Benchmark CUTLASS swiglu7-fused matmul vs torch eager vs torch.compile.

Computation (canonical swiglu7 projection):

    D = swiglu7(A @ B.T)

    A : (M, K)   bf16
    B : (N, K)   bf16   (torch.nn.Linear weight convention; N must be even)
    D : (M, N/2) bf16

where, in pure PyTorch:

    def swiglu7(x, alpha=1.702, limit=7.0):
        x = x.to(float32)
        x_glu, x_linear = x[..., 0::2], x[..., 1::2]
        x_glu    = x_glu.clamp(max=limit)
        x_linear = x_linear.clamp(min=-limit, max=limit)
        out_glu  = x_glu * sigmoid(alpha * x_glu)
        return out_glu * (x_linear + 1.0)

CUTLASS implementation: a single launcher fires two back-to-back Sm80 bf16
GEMMs sharing the same B buffer through strided views (ldB = 2K) — stage 1
precomputes the linear path A @ W_linear.T, stage 2 runs A @ W_gate.T with a
fused EVT epilogue applying SiLU_alpha / clamps / multiply.

Comparison paths
----------------
  eager   : torch.nn.functional.linear(A, B) followed by separate pointwise
            ops (multiple kernel launches, several MxN temporaries).
  compile : torch.compile(eager_ref, mode="max-autotune") — inductor may fuse
            the pointwise tail but cannot merge it back into the GEMM.
  cutlass : two-stage CUTLASS 2.x Sm80EVT kernel on sm_120 (backward-
            compatible Ampere TensorOp running on Blackwell Geforce).

Usage
-----
    python examples/99_evt_demo/swiglu7_epi_pybind_benchmark.py
    python examples/99_evt_demo/swiglu7_epi_pybind_benchmark.py \\
        --shapes "2048,4096,4096;4096,8192,4096" --iters 50
    python examples/99_evt_demo/swiglu7_epi_pybind_benchmark.py --no-compile
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
    (7697, 27304, 5120)
]

ALPHA = 1.702
LIMIT = 7.0


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
        name="swiglu7_epi_torch_ext",
        sources=[os.path.join(HERE, "swiglu7_epi_torch_ext.cu")],
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

def swiglu7_ref(A: torch.Tensor,
                B: torch.Tensor,
                alpha: float = ALPHA,
                limit: float = LIMIT) -> torch.Tensor:
    """Pure PyTorch equivalent of the CUTLASS kernel: D = swiglu7(A @ B.T).

    B is in torch.nn.Linear weight shape (N, K) with N even. The matmul output
    Y = A @ B.T has shape (M, N); even-indexed columns are the gate path,
    odd-indexed columns are the linear path. The matmul accumulates via cuBLAS
    in bf16; we upcast to fp32 for the epilogue to mirror CUTLASS (fp32
    accumulator + fp32 epilogue compute).
    """
    Y = torch.nn.functional.linear(A, B).float()        # (M, N) fp32
    x_glu, x_linear = Y[..., 0::2], Y[..., 1::2]        # each (M, N/2)
    x_glu    = x_glu.clamp(max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu  = x_glu * torch.sigmoid(alpha * x_glu)
    return (out_glu * (x_linear + 1.0)).to(torch.bfloat16)


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

    print(f"Building CUTLASS swiglu7 EVT extension (arch=sm_{args.arch}) ...",
          flush=True)
    ext = build_extension(verbose=args.verbose_build, arch=args.arch)
    print("  built.", flush=True)

    shapes = parse_shape_list(args.shapes) if args.shapes else DEFAULT_SHAPES

    # torch.compile warmup (JIT happens on first call)
    compiled_fn = None
    if not args.no_compile:
        print("torch.compile(swiglu7_ref) ... (first call will JIT)", flush=True)
        compiled_fn = torch.compile(swiglu7_ref, mode="max-autotune",
                                    dynamic=False)

    # ------------------------------------------------------------------
    # Per-shape CUTLASS correctness validation against the PyTorch reference.
    # Runs once per shape with a freshly-seeded random (A, B); reports the
    # absolute error stats and a pass/fail verdict against a relative-error
    # tolerance suited to bf16 (default 1e-2). torch.compile vs eager check
    # is also done on the first shape only (compile JIT is per-shape).
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
            if N % 2 != 0:
                print(f"({M},{N},{K}): SKIPPED (N odd)", flush=True)
                continue
            torch.manual_seed(0xC07A55 + idx)
            A     = torch.randn((M, K),     dtype=torch.bfloat16, device=dev)
            B     = torch.randn((N, K),     dtype=torch.bfloat16, device=dev)
            D_cut = torch.empty((M, N // 2), dtype=torch.bfloat16, device=dev)

            ext.swiglu7_matmul_out(A, B, D_cut)
            D_ref = swiglu7_ref(A, B)

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

            # torch.compile vs eager — on first shape only
            if idx == 0 and compiled_fn is not None:
                D_cmp = compiled_fn(A, B)
                diff2 = (D_cmp.float() - D_ref.float()).abs()
                print(f"  (torch.compile vs eager: "
                      f"max|diff|={diff2.max().item():.4f})",
                      flush=True)

            del A, B, D_cut, D_ref, diff
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
    print(f"\n[swiglu7 fused matmul epilogue — bf16, alpha={ALPHA}, limit={LIMIT}]")
    print(header)
    print("-" * len(header))

    for M, N, K in shapes:
        if N % 2 != 0:
            print(f"Skipping shape ({M},{N},{K}): N must be even.", flush=True)
            continue
        A = torch.randn((M, K),     dtype=torch.bfloat16, device=dev)
        B = torch.randn((N, K),     dtype=torch.bfloat16, device=dev)
        D = torch.empty((M, N // 2), dtype=torch.bfloat16, device=dev)

        def cutlass_fn():
            ext.swiglu7_matmul_out(A, B, D)

        def eager_fn():
            swiglu7_ref(A, B)

        cutlass_ms = bench(cutlass_fn, args.warmup, args.iters)
        eager_ms   = bench(eager_fn,   args.warmup, args.iters)

        if compiled_fn is not None:
            def compile_fn():
                compiled_fn(A, B)
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
