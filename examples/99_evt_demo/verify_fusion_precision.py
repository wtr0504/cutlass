"""Verify that the CUTLASS EVT fused kernel avoids intermediate bf16 round-trips.

Strategy
--------
Build two PyTorch references that differ ONLY in where precision is kept:

  A) bf16-per-step  : each intermediate is .to(bfloat16), mimicking eager mode
                      where every op lands in DRAM as bf16.
  B) fp32-throughout: the entire epilogue runs in fp32, only the final store
                      casts to bf16.  This is the semantics an ideal fused
                      kernel should implement.

If CUTLASS fusion is genuinely "no intermediate bf16 round-trip", then

        | D_cutlass - D_fp32 |   <<   | D_cutlass - D_bf16step |

We print both diffs (max and mean) so the gap is directly visible.

To make intermediate rounding actually *matter*, we deliberately pick
operand magnitudes that put values into bf16's "gappy" region: SiLU outputs
around ~1, then multiplied by a small scale, then added to an Aux of
moderate magnitude.  Under bf16-per-step those tiny contributions are
quantised out; under fp32-throughout they are preserved.
"""

import os
import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
CUTLASS_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def build_bf16_ext():
    return load(
        name="heavy_epi_torch_ext",
        sources=[os.path.join(HERE, "heavy_epi_torch_ext.cu")],
        extra_include_paths=[
            os.path.join(CUTLASS_ROOT, "include"),
            os.path.join(CUTLASS_ROOT, "tools", "util", "include"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-std=c++17", "-O3", "--expt-relaxed-constexpr",
            "-gencode=arch=compute_120,code=sm_120",
        ],
        verbose=False,
    )


def ref_bf16_per_step(A, B, br, sc, Aux):
    """Reference A: every intermediate is downcast to bf16 (eager behaviour)."""
    x = torch.matmul(A, B).to(torch.bfloat16)                 # after MMA
    x = (x + br).to(torch.bfloat16)                            # after + Bias
    x = torch.nn.functional.silu(x).to(torch.bfloat16)         # after SiLU
    x = (x * sc.unsqueeze(-1)).to(torch.bfloat16)              # after * Scale
    x = (x + Aux).to(torch.bfloat16)                           # after + Aux
    x = torch.tanh(x).to(torch.bfloat16)                       # after tanh
    return x


def ref_fp32_throughout(A, B, br, sc, Aux):
    """Reference B: everything in fp32, single downcast at the very end."""
    A32  = A.float();  B32 = B.float()
    br32 = br.float(); sc32 = sc.float(); Aux32 = Aux.float()
    x = A32 @ B32
    x = x + br32
    x = torch.nn.functional.silu(x)
    x = x * sc32.unsqueeze(-1)
    x = x + Aux32
    x = torch.tanh(x)
    return x.to(torch.bfloat16)


def rms(x):
    return x.float().pow(2).mean().sqrt().item()


def compare(name, D_cutlass, D_ref):
    diff = (D_cutlass.float() - D_ref.float()).abs()
    denom = max(1e-12, D_ref.float().abs().mean().item())
    print(f"  vs {name:<18s}  max|diff|={diff.max().item():.3e}  "
          f"mean|diff|={diff.mean().item():.3e}  "
          f"rel={diff.mean().item()/denom:.2e}")


def main():
    dev = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(dev)}", flush=True)
    ext = build_bf16_ext()

    # Sized modest so fp32 reference fits comfortably.
    M, N, K = 1024, 1024, 1024
    torch.manual_seed(0)

    # Pick magnitudes that expose intermediate bf16 rounding:
    #   * A, B small so that (A@B) stays ~O(1)
    #   * bias small, scale ~1, Aux moderate -- makes the SiLU*Scale
    #     contribution land in bf16's rounding gap next to Aux.
    A   = (torch.randn(M, K, device=dev) * 0.05).to(torch.bfloat16)
    B   = (torch.randn(K, N, device=dev) * 0.05).to(torch.bfloat16)
    br  = (torch.randn(N,    device=dev) * 0.10).to(torch.bfloat16)
    sc  = (torch.randn(M,    device=dev) * 0.02 + 0.05).to(torch.bfloat16)
    Aux = (torch.randn(M, N, device=dev) * 1.50).to(torch.bfloat16)
    D   = torch.empty(M, N, dtype=torch.bfloat16, device=dev)

    ext.heavy_epi_matmul_out(A, B, br, sc, Aux, D)

    D_A = ref_bf16_per_step(A, B, br, sc, Aux)
    D_B = ref_fp32_throughout(A, B, br, sc, Aux)

    print(f"Shape: ({M},{N},{K}), output magnitude rms={rms(D_B):.4f}")
    print("CUTLASS fusion output compared to:")
    compare("bf16-per-step  (A)", D, D_A)
    compare("fp32-throughout(B)", D, D_B)

    # How far apart the two references are from each other -- sets the
    # scale that intermediate bf16 round-tripping introduces.
    gap = (D_A.float() - D_B.float()).abs()
    print(f"Gap between A and B (i.e. the precision penalty that eager pays):")
    print(f"  max|A-B|={gap.max().item():.3e}  mean|A-B|={gap.mean().item():.3e}")

    # Headline verdict
    d_fp32 = (D.float() - D_B.float()).abs().mean().item()
    d_bf16 = (D.float() - D_A.float()).abs().mean().item()
    ratio = d_bf16 / max(d_fp32, 1e-12)
    print()
    print(f"CUTLASS is {ratio:.1f}x closer to the fp32-throughout reference "
          f"than to the bf16-per-step reference.")
    if ratio > 5.0:
        print("=> CUTLASS EVT fusion DOES avoid intermediate bf16 round-trips.")
    else:
        print("=> inconclusive (try adjusting operand magnitudes).")


if __name__ == "__main__":
    main()
