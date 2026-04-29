"""Benchmark: CUTLASS two-chunk uneven epilogue vs torch eager / torch.compile.

Computation:

    D[:,    :N1]      = silu(A @ W1.T)
    D[:, N1:N1+N2]    = tanh(A @ W2.T)

    A  : (M, K)        bf16
    B  : (N1+N2, K)    bf16   (concatenated [W1; W2], a single tensor)
    D  : (M, N1+N2)    bf16

In pure PyTorch:

    def uneven_epi(A, B, N1):
        Y  = (A @ B.T).float()              # (M, N1+N2)
        Y1 = torch.nn.functional.silu(Y[:, :N1])
        Y2 = torch.tanh(Y[:, N1:])
        return torch.cat([Y1, Y2], dim=-1).to(torch.bfloat16)

Why this isn't DualGemm
-----------------------
DualGemm fuses two GEMMs that share A in smem; it requires both GEMMs to
have the SAME problem shape (and the same threadblock tile). Here
N1 != N2 so DualGemm doesn't apply. The CUTLASS kernel here uses TWO
independent GEMM launches, each with its own activation epilogue; A is
streamed from HBM twice instead of once. See uneven_epi_torch_ext.cu for
the trade-off discussion.

Compare-with-this:
  eager   : A @ B.T (one matmul) followed by per-chunk pointwise ops with
            an explicit cat — three+ kernels, one (M, N) intermediate.
  compile : torch.compile(eager_ref, mode="max-autotune") — Inductor will
            usually fuse the silu/tanh/cat tail into a single Triton
            pointwise pass, but the matmul still spills (M, N) to HBM.
  cutlass : two CUTLASS GEMMs each with a fused activation epilogue.
            No HBM intermediate per chunk.

Usage
-----
    python examples/99_evt_demo/uneven_epi_pybind_benchmark.py
    python examples/99_evt_demo/uneven_epi_pybind_benchmark.py \\
        --shapes "4096,3072,4096,2048;8192,5120,4096,3072" --iters 50
    python examples/99_evt_demo/uneven_epi_pybind_benchmark.py --no-compile

Shape format: "M,N1,K,N2;..."  (N = N1 + N2; both N1, N2 must be multiples
of 4 and K a multiple of 8).
"""

import argparse
import os
from typing import Callable, List, Tuple

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# (M, N1, K, N2) — N = N1 + N2
DEFAULT_SHAPES: List[Tuple[int, int, int, int]] = [
    (4096, 3072, 4096, 1024),       # 3:1 split
    (4096, 5120, 4096, 3072),       # ~5:3 split
    (8192, 6144, 4096, 2048),       # 3:1 split, larger
    (2048, 8192, 8192, 2048),       # 4:1 split
    (1024, 10240, 4096, 4096),      # ~5:2 split (e.g., GQA-style fused QKV)
    (7697, 18204, 5120, 9100),      # awkward shapes (matches swiglu7 stress test)
]


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
        name="uneven_epi_torch_ext",
        sources=[os.path.join(HERE, "uneven_epi_torch_ext.cu")],
        extra_include_paths=[
            os.path.join(CUTLASS_ROOT, "include"),
            os.path.join(CUTLASS_ROOT, "tools", "util", "include"),
            HERE,
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Reference implementation (matches the CUTLASS computation exactly)
# ---------------------------------------------------------------------------

def uneven_epi_ref(A: torch.Tensor,
                   B: torch.Tensor,
                   N1: int) -> torch.Tensor:
    Y  = torch.nn.functional.linear(A, B).float()        # (M, N) fp32
    Y1 = torch.nn.functional.silu(Y[:, :N1])
    Y2 = torch.tanh(Y[:, N1:])
    return torch.cat([Y1, Y2], dim=-1).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_shape_list(s: str) -> List[Tuple[int, int, int, int]]:
    out: List[Tuple[int, int, int, int]] = []
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        parts = [int(x) for x in tok.split(",")]
        if len(parts) != 4:
            raise argparse.ArgumentTypeError(
                f"Shape '{tok}' must be M,N1,K,N2 (comma-separated).")
        out.append((parts[0], parts[1], parts[2], parts[3]))
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
    # Total work is the full (M, N, K) matmul — both chunks combined.
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12


def check_alignment(M: int, N1: int, K: int, N2: int) -> str | None:
    if K  % 8 != 0: return f"K={K} not multiple of 8"
    if N1 % 4 != 0: return f"N1={N1} not multiple of 4"
    if N2 % 4 != 0: return f"N2={N2} not multiple of 4"
    return None


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

    print(f"Building CUTLASS uneven-epi extension (arch=sm_{args.arch}) ...",
          flush=True)
    ext = build_extension(verbose=args.verbose_build, arch=args.arch)
    print("  built.", flush=True)

    shapes = parse_shape_list(args.shapes) if args.shapes else DEFAULT_SHAPES

    compiled_fn = None
    if not args.no_compile:
        print("torch.compile(uneven_epi_ref) ... (first call will JIT)",
              flush=True)
        compiled_fn = torch.compile(uneven_epi_ref, mode="max-autotune",
                                    dynamic=False)

    # ------------------------------------------------------------------
    # Per-shape CUTLASS correctness validation against the PyTorch reference.
    # ------------------------------------------------------------------
    if not args.skip_validate:
        REL_TOL = args.rel_tol
        print(f"\n[CUTLASS correctness vs PyTorch reference, rel-tol={REL_TOL:.0e}]",
              flush=True)
        v_header = (
            f"{'shape (M,N1,K,N2)':<24}"
            f" {'max|diff|':>10} {'mean|diff|':>11} {'rel':>10}  verdict"
        )
        print(v_header)
        print("-" * len(v_header))

        all_pass = True
        for idx, (M, N1, K, N2) in enumerate(shapes):
            err = check_alignment(M, N1, K, N2)
            if err:
                print(f"({M},{N1},{K},{N2}): SKIPPED ({err})", flush=True)
                continue
            N = N1 + N2
            torch.manual_seed(0xC07A55 + idx)
            A     = torch.randn((M, K), dtype=torch.bfloat16, device=dev)
            B     = torch.randn((N, K), dtype=torch.bfloat16, device=dev)
            D_cut = torch.empty((M, N), dtype=torch.bfloat16, device=dev)

            ext.uneven_epi_matmul_out(A, B, D_cut, N1)
            D_ref = uneven_epi_ref(A, B, N1)

            diff = (D_cut.float() - D_ref.float()).abs()
            ref_mean_abs = D_ref.float().abs().mean().item()
            rel = (diff.mean() / max(1e-6, ref_mean_abs)).item()
            ok  = rel <= REL_TOL
            all_pass = all_pass and ok
            print(
                f"({M},{N1},{K},{N2}):".ljust(24)
                + f" {diff.max().item():>10.4f}"
                + f" {diff.mean().item():>11.5f}"
                + f" {rel:>10.2e}"
                + ("  PASS" if ok else "  FAIL"),
                flush=True,
            )

            # torch.compile vs eager — on first shape only
            if idx == 0 and compiled_fn is not None:
                D_cmp = compiled_fn(A, B, N1)
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
        f"{'shape (M,N1,K,N2)':<24}"
        f" {'eager ms':>10} {'eager TF':>10}"
        f" {'compile ms':>11} {'compile TF':>11}"
        f" {'CUTLASS ms':>11} {'CUTLASS TF':>11}"
        f" {'vs eager':>9} {'vs compile':>11}"
    )
    print(f"\n[uneven epi (silu | tanh) — bf16, two CUTLASS GEMMs per call]")
    print(header)
    print("-" * len(header))

    for M, N1, K, N2 in shapes:
        err = check_alignment(M, N1, K, N2)
        if err:
            print(f"Skipping ({M},{N1},{K},{N2}): {err}.", flush=True)
            continue
        N = N1 + N2
        A = torch.randn((M, K), dtype=torch.bfloat16, device=dev)
        B = torch.randn((N, K), dtype=torch.bfloat16, device=dev)
        D = torch.empty((M, N), dtype=torch.bfloat16, device=dev)

        def cutlass_fn():
            ext.uneven_epi_matmul_out(A, B, D, N1)

        def eager_fn():
            uneven_epi_ref(A, B, N1)

        cutlass_ms = bench(cutlass_fn, args.warmup, args.iters)
        eager_ms   = bench(eager_fn,   args.warmup, args.iters)

        if compiled_fn is not None:
            def compile_fn():
                compiled_fn(A, B, N1)
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
            f"({M},{N1},{K},{N2}):".ljust(24)
            + f" {eager_ms:>10.3f} {eager_tf:>10.1f}"
            + f" {compile_ms_str} {compile_tf_str}"
            + f" {cutlass_ms:>11.3f} {cutlass_tf:>11.1f}"
            + f" {speedup_e:>8.2f}x"
            + f" {speedup_c_str}",
            flush=True,
        )


if __name__ == "__main__":
    main()
