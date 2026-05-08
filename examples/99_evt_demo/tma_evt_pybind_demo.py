"""Minimum driver for the TMA + EVT demo extension (Sm120 / RTX 5090).

Builds `tma_evt_torch_ext.cu`, runs the fused op once, and compares against a
PyTorch reference. The fused op is:

    D = ReLU(A @ B + alpha * C + bias_row)

A and B are FP8 E4M3, packed as torch.uint8 (1 byte per element).
C, bias_row, D are bf16. Accumulation is fp32; epilogue compute is fp32.

The CUTLASS kernel uses TMA for A/B (Sm120 mainloop), TMA for C and D
(epilogue), and a small Sm90 EVT for the elementwise tail (Sm90 EVT visitors
are reused on Sm120 — that's why the EVT type names start with `Sm90`).

Hardware: Blackwell-consumer (sm_120, RTX 5090). Builds for `compute_120a`
(the trailing 'a' unlocks the F8F6F4 Tensor Core MMA — plain `compute_120`
compiles but device-asserts at launch).

Usage:
    python examples/99_evt_demo/tma_evt_pybind_demo.py
    python examples/99_evt_demo/tma_evt_pybind_demo.py --shape 4096,4096,4096
"""

import argparse
import os

import torch
from torch.utils.cpp_extension import load


HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def build_extension(verbose: bool = False):
    return load(
        name="tma_evt_torch_ext",
        sources=[os.path.join(HERE, "tma_evt_torch_ext.cu")],
        extra_include_paths=[
            os.path.join(CUTLASS_ROOT, "include"),
            os.path.join(CUTLASS_ROOT, "tools", "util", "include"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-std=c++17",
            "-O3",
            "--expt-relaxed-constexpr",
            # 'a' suffix is mandatory for the Sm120 F8F6F4 MMA arch gate.
            "-gencode=arch=compute_120a,code=sm_120a",
        ],
        verbose=verbose,
    )


def reference(A_e4m3: torch.Tensor, B_e4m3: torch.Tensor,
              C: torch.Tensor, bias_row: torch.Tensor,
              alpha: float) -> torch.Tensor:
    """fp32 reference — upcast FP8 inputs to fp32, do the matmul, then the EVT."""
    A_f = A_e4m3.to(torch.float32)
    B_f = B_e4m3.to(torch.float32)
    x = A_f @ B_f                           # (M, N) fp32
    x = x + alpha * C.float()
    x = x + bias_row.float()
    return torch.relu(x).to(torch.bfloat16)


def parse_shape(s: str):
    parts = [int(t) for t in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--shape must be M,N,K")
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", type=parse_shape, default=(2048, 2048, 2048))
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--verbose-build", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    dev = torch.device("cuda:0")
    major, minor = torch.cuda.get_device_capability(dev)
    print(f"Device: {torch.cuda.get_device_name(dev)} (sm_{major}{minor})", flush=True)
    if (major, minor) != (12, 0):
        print("WARNING: this demo targets sm_120a (Blackwell-consumer / RTX 5090). "
              "Other devices will likely fail at load or launch.", flush=True)

    M, N, K = args.shape
    assert M % 128 == 0 and N % 128 == 0 and K % 16 == 0, (
        f"FP8 builder requires M%128=N%128=0 and K%16=0; got ({M},{N},{K})")

    print("Building tma_evt_torch_ext (compute_120a) ...", flush=True)
    ext = build_extension(verbose=args.verbose_build)
    print("  built.", flush=True)

    g = torch.Generator(device=dev).manual_seed(0)

    # FP8 A in (M, K) RowMajor  → byte-flatten directly
    A_e4m3 = (torch.randn((M, K), dtype=torch.float32, device=dev, generator=g) * 0.1
              ).to(torch.float8_e4m3fn)
    # FP8 B in (K, N) logical  but the kernel wants ColumnMajor (TN).
    # ColumnMajor (K, N) ≡ contiguous (N, K) byte storage; we generate (N, K)
    # in fp32 and view its byte buffer flat.
    B_e4m3_nk = (torch.randn((N, K), dtype=torch.float32, device=dev, generator=g) * 0.1
                 ).to(torch.float8_e4m3fn)

    # Equivalent logical (K, N) FP8 view of B for the reference matmul.
    B_e4m3_kn = B_e4m3_nk.t().contiguous() if False else B_e4m3_nk.t()

    A_bytes = A_e4m3.contiguous().view(torch.uint8).reshape(-1)
    B_bytes = B_e4m3_nk.contiguous().view(torch.uint8).reshape(-1)

    C   = torch.randn((M, N), dtype=torch.bfloat16, device=dev, generator=g) * 0.1
    br  = torch.randn((N,),   dtype=torch.bfloat16, device=dev, generator=g) * 0.1
    D   = torch.empty((M, N), dtype=torch.bfloat16, device=dev)

    ext.tma_evt_matmul_out(A_bytes, B_bytes, C, br, args.alpha, D, M, N, K)

    D_ref = reference(A_e4m3, B_e4m3_kn, C, br, args.alpha)

    diff = (D.float() - D_ref.float()).abs()
    rel = diff.mean().item() / max(1e-6, D_ref.float().abs().mean().item())
    print(f"Shape (M,N,K)=({M},{N},{K})  alpha={args.alpha}", flush=True)
    print(f"  max|diff|  = {diff.max().item():.4f}", flush=True)
    print(f"  mean|diff| = {diff.mean().item():.5f}", flush=True)
    print(f"  rel        = {rel:.2e}", flush=True)


if __name__ == "__main__":
    main()
