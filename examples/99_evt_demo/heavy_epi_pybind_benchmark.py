"""Benchmark CUTLASS EVT heavy-epilogue matmul across multiple A/B dtypes.

Fused op (identical for every variant):

    D = tanh( SiLU(A @ B + bias_row) * scale_col + Aux )

Dtype variants
--------------
  * bf16          : A/B are bf16.  THREE kernels are benchmarked side-by-side:
                      - ``heavy_epi_torch_ext.cu``    : CUTLASS 2.x **Sm80**
                        EVT (Ampere-class MMA, runs on sm_120 via backward
                        compat).
                      - ``heavy_epi_90_torch_ext.cu`` : CUTLASS 3.x **Sm90**
                        EVT (CollectiveBuilder + Sm90EVT, built with
                        compute_90a PTX so it JITs onto sm_120 at launch).
                      - ``120_90evt_heavy_epi_torch_ext.cu`` : CUTLASS 3.x
                        **arch::Sm120 + Sm90EVT**.  Sm120 dense MMA is
                        F8F6F4-only, so this column quantises the bf16 A/B
                        to FP8 E4M3 *outside* the timed region and then runs
                        the heavy epilogue on the Sm120 mainloop.
                    The Sm80/Sm90 columns expose ``heavy_epi_matmul_out
                    (A,B,br,sc,Aux,D)``; the Sm120 column adds (M,N,K) and
                    takes packed-uint8 A/B.
  * fp8_e4m3 / fp8_e5m2 / fp6_e3m2 / fp6_e2m3 / fp4_e2m1
                  : A/B are Blackwell narrow-precision types.  Kernel is in
                    ``heavy_epi_low_precision_torch_ext.cu`` (CUTLASS 3.x
                    ``CollectiveBuilder`` + ``Sm90EVT``, targets the sm_120a
                    F8F6F4 Tensor Core MMA).  These variants require the
                    separate ``sm_120a`` (note the ``a``) compile target and
                    pass A/B as packed ``torch.uint8`` buffers.

Epilogue tensors (bias_row, scale_col, Aux, D) stay bf16 in every variant so
all kernels produce directly comparable bf16 outputs.

The torch reference decomposes the epilogue into separate ops.  In eager mode
it materialises several MxN intermediates; with ``torch.compile`` inductor is
free to fuse the elementwise tail but still cannot fuse back into the GEMM.

Usage
-----
    python examples/99_evt_demo/heavy_epi_pybind_benchmark.py
    python examples/99_evt_demo/heavy_epi_pybind_benchmark.py \\
        --shapes "4096,4096,4096;8192,8192,8192" --iters 50
    python examples/99_evt_demo/heavy_epi_pybind_benchmark.py --no-compile
    python examples/99_evt_demo/heavy_epi_pybind_benchmark.py \\
        --dtypes "bf16,fp8_e4m3,fp4_e2m1"
"""

import argparse
import os
from typing import Callable, Dict, List, Tuple

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
]

# Describes every low-precision variant:
#   (display_name, extension_entry_point, bits_per_element)
LOW_PRECISION_VARIANTS: List[Tuple[str, str, int]] = [
    ("fp8_e4m3", "heavy_epi_matmul_fp8_e4m3", 8),
    ("fp8_e5m2", "heavy_epi_matmul_fp8_e5m2", 8),
    ("fp6_e3m2", "heavy_epi_matmul_fp6_e3m2", 6),
    ("fp6_e2m3", "heavy_epi_matmul_fp6_e2m3", 6),
    ("fp4_e2m1", "heavy_epi_matmul_fp4_e2m1", 4),
]


# ---------------------------------------------------------------------------
# Extension builders
# ---------------------------------------------------------------------------

def build_extension_bf16(verbose: bool = False, arch: str = "120"):
    """Build the bf16 Sm80-EVT extension (CUTLASS 2.x path)."""
    extra_cuda_cflags = [
        "-std=c++17",
        "-O3",
        "--expt-relaxed-constexpr",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
    ]
    return load(
        name="heavy_epi_torch_ext",
        sources=[os.path.join(HERE, "heavy_epi_torch_ext.cu")],
        extra_include_paths=[
            os.path.join(CUTLASS_ROOT, "include"),
            os.path.join(CUTLASS_ROOT, "tools", "util", "include"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=verbose,
    )


def build_extension_bf16_90(verbose: bool = False):
    """Build the bf16 Sm90-EVT extension (CUTLASS 3.x path).

    ArchTag is Sm90 in the .cu; we emit compute_90a SASS for native Hopper
    plus compute_90a PTX so the runtime can JIT onto sm_120+ Blackwell GPUs.
    The build does NOT depend on the user's ``--arch`` flag — this is fixed
    to compute_90a regardless of the device family.
    """
    extra_cuda_cflags = [
        "-std=c++17",
        "-O3",
        "--expt-relaxed-constexpr",
        "-gencode=arch=compute_120a,code=sm_120a",
    ]
    return load(
        name="heavy_epi_90_torch_ext",
        sources=[os.path.join(HERE, "heavy_epi_90_torch_ext.cu")],
        extra_include_paths=[
            os.path.join(CUTLASS_ROOT, "include"),
            os.path.join(CUTLASS_ROOT, "tools", "util", "include"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=verbose,
    )


def build_extension_bf16_120_90evt(verbose: bool = False, arch: str = "120"):
    """Build the ``arch::Sm120 + Sm90EVT`` heavy-epilogue extension.

    The Sm120 dense F8F6F4 MMA only compiles when targeting ``sm_<arch>a``
    (the ``a`` suffix), same as the multi-dtype low-precision kernel.  A/B
    on the kernel side are FP8 E4M3 packed uint8; the Python bench feeds
    them by quantising bf16 A/B once before the timed region.
    """
    arch_a = f"{arch}a"
    extra_cuda_cflags = [
        "-std=c++17",
        "-O3",
        "--expt-relaxed-constexpr",
        f"-gencode=arch=compute_{arch_a},code=sm_{arch_a}",
    ]
    return load(
        name="heavy_epi_120_90evt_ext",
        sources=[os.path.join(HERE, "120_90evt_heavy_epi_torch_ext.cu")],
        extra_include_paths=[
            os.path.join(CUTLASS_ROOT, "include"),
            os.path.join(CUTLASS_ROOT, "tools", "util", "include"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=verbose,
    )


def build_extension_low_precision(verbose: bool = False, arch: str = "120"):
    """Build the F8F6F4 EVT extension (CUTLASS 3.x path).

    The SM120 F8F6F4 Tensor Core MMA is arch-conditional and compiles only
    when we target ``sm_<arch>a`` (the ``a`` suffix).  Plain ``sm_120``
    compiles but triggers a device-side assert at launch time.
    """
    arch_a = f"{arch}a"
    extra_cuda_cflags = [
        "-std=c++17",
        "-O3",
        "--expt-relaxed-constexpr",
        f"-gencode=arch=compute_{arch_a},code=sm_{arch_a}",
    ]
    return load(
        name="heavy_epi_lp_ext",
        sources=[os.path.join(HERE, "heavy_epi_low_precision_torch_ext.cu")],
        extra_include_paths=[
            os.path.join(CUTLASS_ROOT, "include"),
            os.path.join(CUTLASS_ROOT, "tools", "util", "include"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=verbose,
    )


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


def parse_dtype_list(s: str) -> List[str]:
    all_names = ["bf16"] + [v[0] for v in LOW_PRECISION_VARIANTS]
    if not s:
        return all_names
    names = [t.strip() for t in s.split(",") if t.strip()]
    for n in names:
        if n not in all_names:
            raise argparse.ArgumentTypeError(
                f"Unknown dtype '{n}'. Valid: {all_names}")
    return names


# Matches the CUTLASS EVT kernel mathematically (bf16 I/O, fp32 accum+compute).
def heavy_epi_ref(A: torch.Tensor,
                  B: torch.Tensor,
                  bias_row: torch.Tensor,
                  scale_col: torch.Tensor,
                  Aux: torch.Tensor) -> torch.Tensor:
    x = torch.matmul(A, B)                      # (M, N) bf16 (cuBLAS via torch.mm)
    x = x + bias_row                            # broadcast (N,)
    x = torch.nn.functional.silu(x)
    x = x * scale_col.unsqueeze(-1)             # broadcast (M, 1)
    x = x + Aux
    x = torch.tanh(x)
    return x


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
    """GEMM-only TFLOPS (ignoring the epilogue FLOPs, comparable across paths)."""
    return 2.0 * M * N * K / (ms * 1e-3) / 1e12


def alloc_packed_bytes(numel: int, bits: int,
                       device: torch.device,
                       generator: torch.Generator = None) -> torch.Tensor:
    """Allocate a random uint8 buffer that holds ``numel`` values of ``bits``.

    For a *performance* benchmark the bit pattern is irrelevant: the MMA does
    the same amount of work regardless of the numerical value.  We simply fill
    with random bytes so that the kernel can't shortcut anything.
    """
    total_bits = numel * bits
    assert total_bits % 8 == 0, (
        f"numel*bits must be a multiple of 8 (got {numel}*{bits})")
    total_bytes = total_bits // 8
    return torch.randint(0, 256, (total_bytes,),
                         dtype=torch.uint8, device=device,
                         generator=generator)


def quantize_bf16_to_e4m3_bytes(A_bf16: torch.Tensor,
                                B_bf16: torch.Tensor
                                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert bf16 A (M,K) RowMajor and B (K,N) RowMajor into the byte
    layout the Sm120 F8F6F4 kernel expects: A as (M*K,) uint8 RowMajor and
    B as (K*N,) uint8 ColumnMajor (because the Sm120 builder requires TN).

    Done with PyTorch's native ``float8_e4m3fn`` cast (saturating, RNE on
    GPU); bit-pattern is then re-viewed as uint8 for the C++ side.  This
    runs once per shape, outside the timed region, so it doesn't affect
    benchmark numbers.
    """
    A_fp8 = A_bf16.contiguous().to(torch.float8_e4m3fn)
    A_bytes = A_fp8.view(torch.uint8).reshape(-1).contiguous()
    # B (K,N) RowMajor ->  (N,K) RowMajor (= K-major ColumnMajor over (K,N))
    B_fp8 = B_bf16.contiguous().to(torch.float8_e4m3fn)
    B_bytes = B_fp8.t().contiguous().view(torch.uint8).reshape(-1).contiguous()
    return A_bytes, B_bytes


def alignment_ok(M: int, N: int, K: int, bits: int) -> bool:
    """Shape alignment requirements enforced by the sm_120 F8F6F4 builder.

    ``cutlass::detail::get_input_alignment_bits`` requires at least 128 bits
    on FP8 and 96 B / 64 B on FP6 / FP4 respectively, which translates to K
    being a multiple of 16 (FP8) or 128 (FP6/FP4) elements.  The 128x128
    TileShape adds the usual M%128==0 and N%128==0 constraints.
    """
    if M % 128 or N % 128:
        return False
    if bits == 8:
        return K % 16 == 0
    return K % 128 == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", type=str, default="")
    ap.add_argument("--dtypes", type=str, default="",
                    help="Comma list; subset of "
                         "bf16,fp8_e4m3,fp8_e5m2,fp6_e3m2,fp6_e2m3,fp4_e2m1")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--arch", type=str, default="120")
    ap.add_argument("--verbose-build", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--no-compile", action="store_true",
                    help="Skip torch.compile benchmarking")
    ap.add_argument("--no-sm90", action="store_true",
                    help="Skip the Sm90-EVT bf16 extension (build + bench)")
    ap.add_argument("--force-sm90", action="store_true",
                    help="Force-build the Sm90-EVT extension on non-Hopper devices "
                         "(launches will device-side-assert in WGMMA stubs)")
    ap.add_argument("--no-sm120", action="store_true",
                    help="Skip the arch::Sm120 + Sm90EVT FP8-E4M3 extension "
                         "(build + bench)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    dev = torch.device("cuda:0")
    major, minor = torch.cuda.get_device_capability(dev)
    print(f"Device: {torch.cuda.get_device_name(dev)} (sm_{major}{minor})", flush=True)
    print(f"torch: {torch.__version__}", flush=True)

    dtypes = parse_dtype_list(args.dtypes)
    need_bf16 = "bf16" in dtypes
    low_prec = [v for v in LOW_PRECISION_VARIANTS if v[0] in dtypes]

    # The Sm90-EVT bf16 extension uses WGMMA, which CUTLASS gates strictly on
    # __CUDA_ARCH__ == 900 — Blackwell-consumer (sm_120) cubins compile but
    # the WGMMA paths become device-side assertion stubs. Auto-skip on any
    # device that isn't Hopper unless the user explicitly forces it.
    is_hopper = (major == 9)
    want_sm90 = need_bf16 and not args.no_sm90 and (is_hopper or args.force_sm90)
    if need_bf16 and not args.no_sm90 and not is_hopper and not args.force_sm90:
        print(f"NOTE: skipping Sm90-EVT bf16 extension — device is sm_{major}{minor}, "
              f"WGMMA mainloop requires sm_90 (H100). Pass --force-sm90 to attempt anyway.",
              flush=True)

    # Sm120 + Sm90EVT path requires sm_120 (or higher) for the F8F6F4 MMA.
    # Auto-skip on non-Blackwell-consumer parts; the user can still bench
    # Sm80/Sm90 columns alone.
    is_blackwell_consumer = (major == 12)
    want_sm120 = need_bf16 and not args.no_sm120 and is_blackwell_consumer
    if need_bf16 and not args.no_sm120 and not is_blackwell_consumer:
        print(f"NOTE: skipping arch::Sm120 + Sm90EVT extension — device is sm_{major}{minor}, "
              f"F8F6F4 dense MMA requires sm_120 (Blackwell-consumer).",
              flush=True)

    ext_bf16 = None
    ext_bf16_90 = None
    ext_bf16_120_90evt = None
    ext_lp = None
    if need_bf16:
        print(f"Building CUTLASS bf16 Sm80-EVT extension (arch=sm_{args.arch}) ...", flush=True)
        ext_bf16 = build_extension_bf16(verbose=args.verbose_build, arch=args.arch)
        print("  built.", flush=True)
    if want_sm90:
        print("Building CUTLASS bf16 Sm90-EVT extension (sm_90a + sm_120a) ...", flush=True)
        ext_bf16_90 = build_extension_bf16_90(verbose=args.verbose_build)
        print("  built.", flush=True)
    if want_sm120:
        print(f"Building CUTLASS arch::Sm120 + Sm90EVT FP8-E4M3 extension "
              f"(arch=sm_{args.arch}a) ...", flush=True)
        ext_bf16_120_90evt = build_extension_bf16_120_90evt(
            verbose=args.verbose_build, arch=args.arch)
        print("  built.", flush=True)
    if low_prec:
        print(f"Building CUTLASS F8F6F4 EVT extension (arch=sm_{args.arch}a) ...", flush=True)
        ext_lp = build_extension_low_precision(verbose=args.verbose_build, arch=args.arch)
        print("  built.", flush=True)

    shapes = parse_shape_list(args.shapes) if args.shapes else DEFAULT_SHAPES

    # -- torch.compile warmup (only needed for bf16 reference path) --
    compiled_fn = None
    if need_bf16 and not args.no_compile:
        print("torch.compile(heavy_epi_ref) ... (first call will JIT)", flush=True)
        compiled_fn = torch.compile(heavy_epi_ref, mode="max-autotune", dynamic=False)

    # -- Correctness sanity (bf16 path only; low precision is benched with
    #    random bytes so we just verify that launches don't error out) --
    if need_bf16 and not args.skip_validate:
        M0, N0, K0 = shapes[0]
        A  = torch.randn((M0, K0), dtype=torch.bfloat16, device=dev)
        B  = torch.randn((K0, N0), dtype=torch.bfloat16, device=dev)
        br = torch.randn((N0,),    dtype=torch.bfloat16, device=dev) * 0.1
        sc = torch.randn((M0,),    dtype=torch.bfloat16, device=dev) * 0.5 + 1.0
        Aux = torch.randn((M0, N0), dtype=torch.bfloat16, device=dev) * 0.1
        D_cutlass = torch.empty((M0, N0), dtype=torch.bfloat16, device=dev)
        ext_bf16.heavy_epi_matmul_out(A, B, br, sc, Aux, D_cutlass)
        D_ref = heavy_epi_ref(A, B, br, sc, Aux)
        diff = (D_cutlass.float() - D_ref.float()).abs()
        rel = (diff.mean() / max(1e-6, D_ref.float().abs().mean().item())).item()
        print(
            f"Correctness (bf16 Sm80-EVT) on ({M0},{N0},{K0}): "
            f"max|diff|={diff.max().item():.4f}, "
            f"mean|diff|={diff.mean().item():.5f}, rel={rel:.2e}",
            flush=True,
        )
        if want_sm90:
            D_cutlass90 = torch.empty((M0, N0), dtype=torch.bfloat16, device=dev)
            ext_bf16_90.heavy_epi_matmul_out(A, B, br, sc, Aux, D_cutlass90)
            diff90 = (D_cutlass90.float() - D_ref.float()).abs()
            rel90 = (diff90.mean() / max(1e-6, D_ref.float().abs().mean().item())).item()
            print(
                f"Correctness (bf16 Sm90-EVT) on ({M0},{N0},{K0}): "
                f"max|diff|={diff90.max().item():.4f}, "
                f"mean|diff|={diff90.mean().item():.5f}, rel={rel90:.2e}",
                flush=True,
            )
        if want_sm120:
            # FP8 E4M3 quantisation introduces real numeric error vs bf16
            # reference (E4M3 has ~3 mantissa bits); print it for visibility.
            A_bytes0, B_bytes0 = quantize_bf16_to_e4m3_bytes(A, B)
            D_cutlass120 = torch.empty((M0, N0), dtype=torch.bfloat16, device=dev)
            ext_bf16_120_90evt.heavy_epi_matmul_out(
                A_bytes0, B_bytes0, br, sc, Aux, D_cutlass120, M0, N0, K0)
            diff120 = (D_cutlass120.float() - D_ref.float()).abs()
            rel120 = (diff120.mean() / max(1e-6, D_ref.float().abs().mean().item())).item()
            print(
                f"Correctness (Sm120 + Sm90EVT, FP8 E4M3 A/B) on ({M0},{N0},{K0}): "
                f"max|diff|={diff120.max().item():.4f}, "
                f"mean|diff|={diff120.mean().item():.5f}, rel={rel120:.2e} "
                f"(quantisation noise expected)",
                flush=True,
            )
        if compiled_fn is not None:
            D_cmp = compiled_fn(A, B, br, sc, Aux)
            diff2 = (D_cmp.float() - D_ref.float()).abs()
            print(f"  torch.compile vs eager: max|diff|={diff2.max().item():.4f}",
                  flush=True)

    # Smoke-test every low-precision variant on the smallest tile (128x128x128)
    # with all-zero A/B, where the expected output collapses to
    # tanh(SiLU(bias) * scale + aux) and should match to within bf16 rounding.
    if low_prec and not args.skip_validate:
        print("Low-precision correctness (A=B=0, 128x128x128):", flush=True)
        M0 = N0 = K0 = 128
        br = torch.randn((N0,),    dtype=torch.bfloat16, device=dev) * 0.1
        sc = torch.randn((M0,),    dtype=torch.bfloat16, device=dev) * 0.5 + 1.0
        Aux = torch.randn((M0, N0), dtype=torch.bfloat16, device=dev) * 0.1
        D_ref_lp = torch.tanh(
            torch.nn.functional.silu(br.float().expand(M0, N0))
            * sc.float().unsqueeze(-1)
            + Aux.float()
        ).to(torch.bfloat16)
        for name, entry, bits in low_prec:
            A_bytes = torch.zeros(M0 * K0 * bits // 8,
                                  dtype=torch.uint8, device=dev)
            B_bytes = torch.zeros(K0 * N0 * bits // 8,
                                  dtype=torch.uint8, device=dev)
            D = torch.empty((M0, N0), dtype=torch.bfloat16, device=dev)
            getattr(ext_lp, entry)(A_bytes, B_bytes, br, sc, Aux, D,
                                   M0, N0, K0)
            diff = (D.float() - D_ref_lp.float()).abs()
            print(f"  {name:<10s} max|diff|={diff.max().item():.4e} "
                  f"mean|diff|={diff.mean().item():.4e}",
                  flush=True)

    # -- Benchmarks --
    # Table 1: bf16 comparison.  Adds a CUTLASS-Sm90 column when --no-sm90
    # was not passed, so the user can read off Sm80-EVT vs Sm90-EVT side by
    # side at every shape.  vs-eager / vs-compile speedups are reported for
    # the Sm80 column (the historical baseline); the Sm90 row also prints
    # its speedup vs Sm80 so the EVT-version-only delta is obvious.
    if need_bf16:
        header_cells = [
            f"{'shape (M,N,K)':<22}",
            f" {'eager ms':>10} {'eager TF':>10}",
            f" {'compile ms':>11} {'compile TF':>11}",
            f" {'CUT-80 ms':>11} {'CUT-80 TF':>11}",
        ]
        if want_sm90:
            header_cells.append(f" {'CUT-90 ms':>11} {'CUT-90 TF':>11}")
        if want_sm120:
            # Sm120 path takes FP8 A/B but the user-facing inputs are bf16,
            # so we report three numbers per shape:
            #   kern ms : pre-quantised inputs, kernel only
            #   Q ms    : bf16 -> FP8 E4M3 conversion of *both* A and B
            #   +Q ms   : kern + Q (the apples-to-apples cost vs the bf16
            #             baseline, which would also start from bf16 A/B)
            # +Q TF is the effective throughput counting the quant cost.
            header_cells.append(
                f" {'CUT-120 kern':>13} {'Q ms':>9} {'CUT-120 +Q ms':>15} {'+Q TF':>9}")
        header_cells += [
            f" {'vs eager':>9} {'vs compile':>11}",
        ]
        if want_sm90:
            header_cells.append(f" {'90 vs 80':>9}")
        if want_sm120:
            header_cells.append(f" {'120+Q vs 80':>12}")
        header = "".join(header_cells)
        title_bits = ["Sm80-EVT"]
        if want_sm90:
            title_bits.append("Sm90-EVT")
        if want_sm120:
            title_bits.append("Sm120+Sm90EVT(FP8E4M3, +bf16->FP8 quant)")
        print(f"\n[bf16 heavy-epilogue GEMM — {' vs '.join(title_bits)}]"
              if (want_sm90 or want_sm120) else "\n[bf16 heavy-epilogue GEMM]")
        if want_sm120:
            print("  Sm120 columns: kern = kernel only (pre-quantised inputs); "
                  "Q = bf16->FP8 of A and B; +Q = kern+Q (fair vs Sm80 bf16)."
                  "\n  +Q assumes B is quantised per call too; in real LLM "
                  "inference weights can be cached and only A pays Q.",
                  flush=True)
        print(header)
        print("-" * len(header))

        for M, N, K in shapes:
            A   = torch.randn((M, K), dtype=torch.bfloat16, device=dev)
            B   = torch.randn((K, N), dtype=torch.bfloat16, device=dev)
            br  = torch.randn((N,),   dtype=torch.bfloat16, device=dev) * 0.1
            sc  = torch.randn((M,),   dtype=torch.bfloat16, device=dev) * 0.5 + 1.0
            Aux = torch.randn((M, N), dtype=torch.bfloat16, device=dev) * 0.1
            D   = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
            D90 = (torch.empty((M, N), dtype=torch.bfloat16, device=dev)
                   if want_sm90 else None)
            D120 = (torch.empty((M, N), dtype=torch.bfloat16, device=dev)
                    if want_sm120 else None)
            # Sm120 path consumes packed FP8 bytes.  Pre-quantise once for
            # the kernel-only timing, but ALSO time the bf16->FP8 conversion
            # (timed inside cutlass120_quant_fn / cutlass120_full_fn below)
            # so the comparison vs the bf16 Sm80 column is apples-to-apples.
            bench_sm120 = want_sm120 and alignment_ok(M, N, K, bits=8)
            if bench_sm120:
                A_bytes_120, B_bytes_120 = quantize_bf16_to_e4m3_bytes(A, B)
            else:
                A_bytes_120 = B_bytes_120 = None

            def cutlass_fn():
                ext_bf16.heavy_epi_matmul_out(A, B, br, sc, Aux, D)

            def eager_fn():
                heavy_epi_ref(A, B, br, sc, Aux)

            cutlass_ms = bench(cutlass_fn, args.warmup, args.iters)
            eager_ms   = bench(eager_fn,   args.warmup, args.iters)

            if compiled_fn is not None:
                def compile_fn():
                    compiled_fn(A, B, br, sc, Aux)
                compile_ms = bench(compile_fn, args.warmup, args.iters)
            else:
                compile_ms = float("nan")

            if want_sm90:
                def cutlass90_fn():
                    ext_bf16_90.heavy_epi_matmul_out(A, B, br, sc, Aux, D90)
                cutlass90_ms = bench(cutlass90_fn, args.warmup, args.iters)
            else:
                cutlass90_ms = float("nan")

            if bench_sm120:
                # Kernel-only: pre-quantised buffers reused across iters.
                def cutlass120_kern_fn(_A=A_bytes_120, _B=B_bytes_120, _D=D120):
                    ext_bf16_120_90evt.heavy_epi_matmul_out(
                        _A, _B, br, sc, Aux, _D, M, N, K)
                cutlass120_kern_ms = bench(cutlass120_kern_fn, args.warmup, args.iters)

                # Quant-only: bf16 -> FP8 E4M3 of A and B, no kernel.  This
                # is the cost the Sm80 baseline does NOT pay (it consumes
                # bf16 directly).
                def cutlass120_quant_fn(_A=A, _B=B):
                    quantize_bf16_to_e4m3_bytes(_A, _B)
                cutlass120_q_ms = bench(cutlass120_quant_fn, args.warmup, args.iters)

                # Full pipeline: fresh quant + kernel each iter.  This is
                # the apples-to-apples cost vs the bf16 Sm80 column.
                def cutlass120_full_fn(_A=A, _B=B, _D=D120):
                    a_b, b_b = quantize_bf16_to_e4m3_bytes(_A, _B)
                    ext_bf16_120_90evt.heavy_epi_matmul_out(
                        a_b, b_b, br, sc, Aux, _D, M, N, K)
                cutlass120_full_ms = bench(cutlass120_full_fn, args.warmup, args.iters)
            else:
                cutlass120_kern_ms = float("nan")
                cutlass120_q_ms    = float("nan")
                cutlass120_full_ms = float("nan")

            cutlass_tf         = gemm_tflops(cutlass_ms,   M, N, K)
            eager_tf           = gemm_tflops(eager_ms,     M, N, K)
            compile_tf         = (gemm_tflops(compile_ms, M, N, K)
                                  if compile_ms == compile_ms else float("nan"))
            cutlass90_tf       = (gemm_tflops(cutlass90_ms, M, N, K)
                                  if cutlass90_ms == cutlass90_ms else float("nan"))
            cutlass120_full_tf = (gemm_tflops(cutlass120_full_ms, M, N, K)
                                  if cutlass120_full_ms == cutlass120_full_ms
                                  else float("nan"))

            speedup_e   = eager_ms   / cutlass_ms
            speedup_c   = (compile_ms / cutlass_ms
                           if compile_ms == compile_ms else float("nan"))
            speedup_90  = (cutlass_ms / cutlass90_ms
                           if cutlass90_ms == cutlass90_ms else float("nan"))
            # Speedup vs Sm80 uses the FULL (kern+Q) cost — this is what an
            # actual replacement of the bf16 Sm80 path would pay.
            speedup_120 = (cutlass_ms / cutlass120_full_ms
                           if cutlass120_full_ms == cutlass120_full_ms
                           else float("nan"))

            row = (
                f"({M},{N},{K}):".ljust(22)
                + f" {eager_ms:>10.3f} {eager_tf:>10.1f}"
                + (f" {compile_ms:>11.3f} {compile_tf:>11.1f}"
                   if compile_ms == compile_ms else
                   f" {'n/a':>11} {'n/a':>11}")
                + f" {cutlass_ms:>11.3f} {cutlass_tf:>11.1f}"
            )
            if want_sm90:
                row += (f" {cutlass90_ms:>11.3f} {cutlass90_tf:>11.1f}"
                        if cutlass90_ms == cutlass90_ms else
                        f" {'n/a':>11} {'n/a':>11}")
            if want_sm120:
                if cutlass120_full_ms == cutlass120_full_ms:
                    row += (f" {cutlass120_kern_ms:>13.3f}"
                            f" {cutlass120_q_ms:>9.3f}"
                            f" {cutlass120_full_ms:>15.3f}"
                            f" {cutlass120_full_tf:>9.1f}")
                else:
                    row += (f" {'n/a':>13} {'n/a':>9} {'n/a':>15} {'n/a':>9}")
            row += (
                f" {speedup_e:>8.2f}x"
                + (f" {speedup_c:>10.2f}x"
                   if speedup_c == speedup_c else f" {'n/a':>11}")
            )
            if want_sm90:
                row += (f" {speedup_90:>8.2f}x"
                        if speedup_90 == speedup_90 else f" {'n/a':>9}")
            if want_sm120:
                row += (f" {speedup_120:>11.2f}x"
                        if speedup_120 == speedup_120 else f" {'n/a':>12}")
            print(row, flush=True)

    # Table 2: Low-precision CUTLASS comparison.  For every shape we time
    # each variant and print its ms / TFLOPS plus speedup vs CUTLASS bf16
    # (when available).  Shapes that don't satisfy the F8F6F4 alignment
    # requirement are skipped with an n/a placeholder.
    if low_prec:
        name_cols = "".join(f" {v[0]:>10}" for v in low_prec)
        tf_cols   = "".join(f" {(v[0]+' TF'):>13}" for v in low_prec)
        bf16_col  = " CUTLASS-bf16" if need_bf16 else ""
        print("\n[low-precision heavy-epilogue GEMM]  time in ms")
        print(f"{'shape (M,N,K)':<22}{bf16_col}{name_cols}")
        print("-" * (22 + len(bf16_col) + len(name_cols)))

        for M, N, K in shapes:
            # Reuse the epilogue tensors so every variant sees the same
            # bias/scale/Aux (so that relative numbers reflect MMA dtype only).
            br  = torch.randn((N,),   dtype=torch.bfloat16, device=dev) * 0.1
            sc  = torch.randn((M,),   dtype=torch.bfloat16, device=dev) * 0.5 + 1.0
            Aux = torch.randn((M, N), dtype=torch.bfloat16, device=dev) * 0.1
            D   = torch.empty((M, N), dtype=torch.bfloat16, device=dev)

            cutlass_bf16_ms = float("nan")
            if need_bf16:
                A = torch.randn((M, K), dtype=torch.bfloat16, device=dev)
                B = torch.randn((K, N), dtype=torch.bfloat16, device=dev)
                def _bf16_fn():
                    ext_bf16.heavy_epi_matmul_out(A, B, br, sc, Aux, D)
                cutlass_bf16_ms = bench(_bf16_fn, args.warmup, args.iters)

            variant_ms: Dict[str, float] = {}
            for name, entry, bits in low_prec:
                if not alignment_ok(M, N, K, bits):
                    variant_ms[name] = float("nan")
                    continue
                fn_handle = getattr(ext_lp, entry)
                A_bytes = alloc_packed_bytes(M * K, bits, dev)
                B_bytes = alloc_packed_bytes(K * N, bits, dev)

                def _lp_fn(fn=fn_handle, A=A_bytes, B=B_bytes):
                    fn(A, B, br, sc, Aux, D, M, N, K)

                variant_ms[name] = bench(_lp_fn, args.warmup, args.iters)

            cells = []
            if need_bf16:
                cells.append(f" {cutlass_bf16_ms:>12.3f}"
                             if cutlass_bf16_ms == cutlass_bf16_ms
                             else f" {'n/a':>12}")
            for name, _, _ in low_prec:
                ms = variant_ms[name]
                cells.append(f" {ms:>10.3f}" if ms == ms else f" {'n/a':>10}")
            print(f"({M},{N},{K}):".ljust(22) + "".join(cells), flush=True)

        # Same table, expressed in TFLOPS.
        print("\n[low-precision heavy-epilogue GEMM]  TFLOPS (2*M*N*K / t)")
        tf_cols2 = "".join(f" {v[0]:>10}" for v in low_prec)
        bf16_col2 = " CUTLASS-bf16" if need_bf16 else ""
        print(f"{'shape (M,N,K)':<22}{bf16_col2}{tf_cols2}")
        print("-" * (22 + len(bf16_col2) + len(tf_cols2)))

        for M, N, K in shapes:
            # Re-time to produce a second, TFLOPS-based view.  We reuse the
            # same alignment check so "n/a" rows stay consistent.
            br  = torch.randn((N,),   dtype=torch.bfloat16, device=dev) * 0.1
            sc  = torch.randn((M,),   dtype=torch.bfloat16, device=dev) * 0.5 + 1.0
            Aux = torch.randn((M, N), dtype=torch.bfloat16, device=dev) * 0.1
            D   = torch.empty((M, N), dtype=torch.bfloat16, device=dev)

            cells = []
            if need_bf16:
                A = torch.randn((M, K), dtype=torch.bfloat16, device=dev)
                B = torch.randn((K, N), dtype=torch.bfloat16, device=dev)
                def _bf16_fn():
                    ext_bf16.heavy_epi_matmul_out(A, B, br, sc, Aux, D)
                cutlass_bf16_ms = bench(_bf16_fn, args.warmup, args.iters)
                cells.append(f" {gemm_tflops(cutlass_bf16_ms, M, N, K):>12.1f}")

            for name, entry, bits in low_prec:
                if not alignment_ok(M, N, K, bits):
                    cells.append(f" {'n/a':>10}")
                    continue
                A_bytes = alloc_packed_bytes(M * K, bits, dev)
                B_bytes = alloc_packed_bytes(K * N, bits, dev)
                fn_handle = getattr(ext_lp, entry)
                def _lp_fn(fn=fn_handle, A=A_bytes, B=B_bytes):
                    fn(A, B, br, sc, Aux, D, M, N, K)
                ms = bench(_lp_fn, args.warmup, args.iters)
                cells.append(f" {gemm_tflops(ms, M, N, K):>10.1f}")

            print(f"({M},{N},{K}):".ljust(22) + "".join(cells), flush=True)


if __name__ == "__main__":
    main()
