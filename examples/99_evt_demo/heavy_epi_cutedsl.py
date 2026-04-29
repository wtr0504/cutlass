# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Same heavy matmul epilogue as ``heavy_epi_torch_ext.cu`` (CUTLASS 2.x EVT),
but expressed with CuTeDSL's **Epilogue Fusion Configuration (EFC)**.

Fused op:

    D = tanh( SiLU(A @ B + Bias) * Scale + Aux )

NOTE (run on RTX 5090 / sm_120?): this file uses EFC and WILL FAIL to run on
sm_120 with:

    OpError: expects arch to be one of [Arch.sm_100a, ..., Arch.sm_110f],
             but got Arch.sm_120a     [MmaF16BF16Op error]

because the EFC framework is wired to ``tcgen05.mma`` (SM100 only).  For an
equivalent **sm_120-compatible** implementation that actually runs on RTX
5090, see ``heavy_epi_sm120_kernel.py`` + ``heavy_epi_sm120_driver.py`` in
this directory -- they take the ``blackwell_geforce/dense_gemm.py`` kernel
and hand-fuse the heavy epilogue at the register stage.

IMPORTANT
---------
* EFC is currently only implemented under
  ``examples/python/CuTeDSL/blackwell/epilogue/`` and targets SM100
  (Blackwell DC, e.g. B100/B200). It relies on tcgen05.mma + TMEM
  + warp-specialized epilogue, which are not available on sm_120 (RTX 5090).
* EFC requires every tensor parameter passed to the epilogue to share the
  layout of D, i.e. (M, N, L). Pure broadcasts like per-column bias (N,)
  or per-row scale (M,) are NOT natively supported; they must be expanded
  to (M, N, L) tensors (see ``create_arguments`` below).

Usage
-----
    cd examples/99_evt_demo
    # needs a B100/B200 to actually run
    python heavy_epi_cutedsl.py \\
        --ab_dtype BFloat16 --c_dtype BFloat16 --aux_dtype BFloat16 --d_dtype BFloat16 \\
        --acc_dtype Float32 --epi_dtype Float32 \\
        --mma_tiler_mn 128,128 --cluster_shape_mn 2,1 \\
        --mnkl 4096,4096,4096,1 --use_2cta_instrs
"""

from __future__ import annotations

import os
import sys
import traceback
import typing

import cuda.bindings.driver as cuda
from typing_extensions import override
import torch

import cutlass
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch

# Point at the reference EFC implementation shipped with CUTLASS.
_HERE = os.path.dirname(os.path.abspath(__file__))
_EFC_DIR = os.path.abspath(
    os.path.join(_HERE, "..", "python", "CuTeDSL", "blackwell", "epilogue")
)
if _EFC_DIR not in sys.path:
    sys.path.insert(0, _EFC_DIR)

from common_dense_gemm_efc import DenseGemmEFC  # noqa: E402


# -----------------------------------------------------------------------------
# THE HEART OF THE EXAMPLE: the EFC epilogue function.
#
# This is the direct CuTeDSL equivalent of this CUTLASS 2.x EVT template tree:
#
#   EVT_D = Sm80EVT<StoreD,
#            Sm80EVT<ComputeTanh,
#              Sm80EVT<ComputeAdd,
#                Sm80EVT<ComputeMul,
#                  Sm80EVT<ComputeSiLu,
#                    Sm80EVT<ComputeAdd, Accum, BiasRow> >,
#                  ScaleCol>,
#                Aux> > >;
#
# Every parameter here (other than ``efc_config``) is either a cute.Tensor
# (recognised by EFC's ParameterAnalysis phase) or a scalar. The first
# parameter MUST be called ``efc_config`` by convention.
# -----------------------------------------------------------------------------
def heavy_epilogue(efc_config, Bias, Scale, Aux, D):
    t0 = efc_config.accum() + Bias.load()   # Accum + Bias
    t1 = efc_config.silu(t0)                 # SiLU(...)
    t2 = t1 * Scale.load()                   # * ScaleCol
    t3 = t2 + Aux.load()                     # + Aux
    D.store(efc_config.tanh(t3))             # tanh(...) -> D


# -----------------------------------------------------------------------------
# Plumbing: wrap the epilogue into a DenseGemmEFC subclass so we can hand it
# to the framework.  This is essentially a trimmed-down copy of
# ``blackwell/epilogue/activation_custom_epilogue_dense_gemm.py``.
# -----------------------------------------------------------------------------
class HeavyEpiGemm(DenseGemmEFC):
    """Dense GEMM with the heavy fused epilogue above."""

    class CLIParser(DenseGemmEFC.CLIParser):
        @override
        def more_parsing(self):
            self.parser.add_argument("--c_dtype", type=cutlass.dtype,
                                     default=cutlass.BFloat16)
            self.parser.add_argument("--aux_dtype", type=cutlass.dtype,
                                     default=cutlass.BFloat16)
            self.parser.add_argument("--d_dtype", type=cutlass.dtype,
                                     default=cutlass.BFloat16)

    @override
    def create_arguments(self, l, m, n, k,
                         a_major, b_major, cd_major,
                         ab_dtype,
                         # supplemental
                         c_dtype, aux_dtype, d_dtype):
        std_args = super().create_arguments(l, m, n, k,
                                            a_major, b_major, cd_major, ab_dtype)

        # Every EFC tensor arg has to share D's (M, N, L) layout, so we
        # materialise the logically-broadcast Bias / Scale into full MxN
        # tensors (initialised from a 1D vector so the math still matches
        # per-column-bias / per-row-scale semantics).
        bias_cpu = cutlass_torch.matrix(l, m, n, cd_major == "m", c_dtype)
        bias_tensor, bias_gpu = cutlass_torch.cute_tensor_like(
            bias_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16)

        scale_cpu = cutlass_torch.matrix(l, m, n, cd_major == "m", c_dtype)
        scale_tensor, scale_gpu = cutlass_torch.cute_tensor_like(
            scale_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16)

        aux_cpu = cutlass_torch.matrix(l, m, n, cd_major == "m", aux_dtype)
        aux_tensor, aux_gpu = cutlass_torch.cute_tensor_like(
            aux_cpu, aux_dtype, is_dynamic_layout=True, assumed_align=16)

        d_cpu = cutlass_torch.matrix(l, m, n, cd_major == "m", d_dtype)
        d_tensor, d_gpu = cutlass_torch.cute_tensor_like(
            d_cpu, d_dtype, is_dynamic_layout=True, assumed_align=16)

        return (*std_args,
                bias_tensor, bias_cpu, bias_gpu,
                scale_tensor, scale_cpu, scale_gpu,
                aux_tensor, aux_cpu, aux_gpu,
                d_tensor, d_cpu, d_gpu)

    def compare(self, a_torch_cpu, b_torch_cpu, epi_dtype, tolerance,
                bias_gpu, scale_gpu, aux_gpu, d_gpu,
                bias_cpu, scale_cpu, aux_cpu, d_cpu):
        # evaluate_on_cpu replays ``heavy_epilogue`` on the CPU in
        # PyTorchEvaluation phase -> writes d_cpu in-place.
        self.evaluate_on_cpu(
            a_torch_cpu, b_torch_cpu, epi_dtype,
            bias_cpu, scale_cpu, aux_cpu, d_cpu)
        torch.testing.assert_close(d_gpu.cpu(), d_cpu, atol=tolerance, rtol=1e-2)
        # sanity: read-only tensors unchanged
        torch.testing.assert_close(bias_gpu.cpu(), bias_cpu, atol=tolerance, rtol=1e-2)
        torch.testing.assert_close(scale_gpu.cpu(), scale_cpu, atol=tolerance, rtol=1e-2)
        torch.testing.assert_close(aux_gpu.cpu(), aux_cpu, atol=tolerance, rtol=1e-2)


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def run(mnkl, ab_dtype, acc_dtype, epi_dtype,
        a_major, b_major, cd_major,
        c_dtype, aux_dtype, d_dtype,
        mma_tiler_mn, cluster_shape_mn, use_2cta_instrs,
        tolerance, warmup_iterations=3, iterations=100, skip_ref_check=False):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required.")
    m, n, k, l = mnkl

    torch_stream  = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    gemm = HeavyEpiGemm(
        acc_dtype, epi_dtype, use_2cta_instrs,
        mma_tiler_mn, cluster_shape_mn,
        heavy_epilogue,   # <-- the epilogue function, traced by EFC
    )

    (a_tensor, b_tensor, a_cpu, b_cpu,
     bias_tensor, bias_cpu, bias_gpu,
     scale_tensor, scale_cpu, scale_gpu,
     aux_tensor, aux_cpu, aux_gpu,
     d_tensor, d_cpu, d_gpu) = gemm.create_arguments(
        l, m, n, k, a_major, b_major, cd_major,
        ab_dtype, c_dtype, aux_dtype, d_dtype)

    gemm.check_implementable(a_tensor, b_tensor, d_tensor)

    max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1])

    # The supplemental args must be passed in the SAME ORDER as in the
    # signature of ``heavy_epilogue`` after efc_config.
    compiled = gemm.compile(
        a_tensor, b_tensor, max_active_clusters, current_stream,
        bias_tensor, scale_tensor, aux_tensor, d_tensor,
    )
    compiled(
        a_tensor, b_tensor, current_stream,
        bias_tensor, scale_tensor, aux_tensor, d_tensor,
    )

    exec_us = testing.benchmark(
        compiled,
        kernel_arguments=testing.JitArguments(
            a_tensor, b_tensor, current_stream,
            bias_tensor, scale_tensor, aux_tensor, d_tensor,
        ),
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )
    print(f"Execution time: {exec_us:.3f} us")

    if not skip_ref_check:
        gemm.compare(a_cpu, b_cpu, epi_dtype, tolerance,
                     bias_gpu, scale_gpu, aux_gpu, d_gpu,
                     bias_cpu, scale_cpu, aux_cpu, d_cpu)
        print("Results match CPU reference.")


if __name__ == "__main__":
    args = HeavyEpiGemm.CLIParser().parse()
    try:
        run(args.mnkl,
            args.ab_dtype, args.acc_dtype, args.epi_dtype,
            args.a_major, args.b_major, args.cd_major,
            args.c_dtype, args.aux_dtype, args.d_dtype,
            args.mma_tiler_mn, args.cluster_shape_mn, args.use_2cta_instrs,
            args.tolerance, args.warmup_iterations, args.iterations,
            args.skip_ref_check)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
