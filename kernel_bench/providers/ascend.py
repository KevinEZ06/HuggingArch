"""The kernels vLLM-Ascend runs on an Ascend NPU, as measurable cells.

Every entry here calls the `torch_npu` entry point the plugin's own source
calls, read at the release the binding is audited at
(`framework_bindings/vllm_ascend.yaml`, v0.30.0). The library is declared
`vllm_ascend` so a row says which stack produced it, and the import gate is
`torch_npu`, which is what those entry points live in.

WHY A SEPARATE MODULE, AND WHY THE PAGE DIFFERS

A paged attention kernel's time depends on the page it reads the cache
through, so the page is an axis of the cell (`KernelSignature`
`kv_block_sizes`). vLLM-Ascend allocates at 128
(`vllm_ascend/utils.py:1504-1505`, read for the binding's `kv_block_size`),
while every CUDA backend here serves 16/32/64. These are therefore two
disjoint sets of cells rather than one set measured twice — which is the
correct outcome: a kernel reading 128-token pages gathers a quarter as many
pages per query as one reading 32, and no row of the one says anything
about the other.

WHAT IS CHECKED ON THE DEVICE, NOT ASSUMED

Each entry point below was called on a 910C die before it was registered,
and the kernel it dispatches to was read off the Ascend profiler rather
than inferred from the function name (`tools/` probe, 2026-11-14):

    torch_npu.npu_add_rms_norm   -> aclnnAddRmsNorm_AddRmsNorm_AddRmsNorm
    torch_npu.npu_swiglu         -> SwiGlu
    torch_npu.npu_dynamic_quant  -> aclnnDynamicQuantV2_DynamicQuant_...
    torch_npu.npu_quant_matmul   -> aclnnQuantMatmulV5_QuantBatchMatmulV3_...

one Ascend kernel per call, three calls per three iterations, with no
PyTorch elementwise chain beside them. That check matters because the
plugin turns its custom ops on through a config switch
(`vllm_ascend/platform.py:1375-1377`, `custom_ops=["all"]`): with it off the
same Python call returns the same numbers from a native torch composition,
and only the dispatched kernel differs. A row measured on that path would
be a correct number for a kernel nobody serves.
"""
from __future__ import annotations

from typing import Any, Dict

from ..invocation import Invocation
from ..env import Env
from . import (BenchCase, KernelSignature, Provider, register)

#: The paged-KV page vLLM-Ascend's attention backends actually run, from
#: the binding (`framework_bindings/vllm_ascend.yaml` `kv_block_size`,
#: read at `vllm_ascend/utils.py:1504-1505`) and from each backend's
#: `get_supported_kernel_block_sizes()` — all `[128]` on A3. Stated once
#: here; a second literal in a build would be a second statement of the
#: kernel page. The hybrid scheduler alignment (640, or 2176/4352 under
#: int8 KV) is the KV-pool grain, not this axis: HuggingArch keys a paged
#: cell on `op_signatures.PAGE_AXIS`, which is the kernel page.
ASCEND_PAGE = 128

#: The float formats these kernels take. The plugin serves a model at its
#: own dtype and the Ascend cube runs bf16 and fp16 at the same rate
#: (`gpu_specs.csv`: both columns 248.4), so both are offered and neither is
#: promoted.
_ASCEND_FLOAT = ("bf16", "fp16")


def _torch_dtype(name: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16,
            "fp32": torch.float32, "int8": torch.int8}[name]


# ---------------------------------------------------------------------------
# W8A8 INT8 — the path `modelslim-w8a8-dynamic` + `int8_per_token` prices.
#
# The activation is quantised per token at run time and one INT8 x INT8
# matmul runs (`vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py:
# 60-131`). Measured on a die: `npu_dynamic_quant` returns int8 elements and
# one FLOAT32 scale per token (shape [tokens]), which is what the activation
# format declares (`arch/activation_formats/int8_per_token.yaml`).
# ---------------------------------------------------------------------------

def _int8_per_channel(w):
    """`w` → `(int8 elements, one scale per output channel)`.

    The harness's own grid (`quant.quantise_groups`, the single
    implementation every quantised cell is placed on), asked for one group
    per row — which is what per-channel means for a `[N, K]` weight. The
    levels come back in the working precision because `int8` declares no
    torch cast, so they are narrowed here; they are already integers, so
    that narrowing is exact.

    Quantising in the BUILD, not in the timed closure: the checkpoint ships
    the weight quantised and the plugin only transposes it at load
    (`w8a8_dynamic.py:136-139`), so this is setup, the same place the fp8
    providers quantise theirs.
    """
    import torch
    from ..quant import FORMATS, quantise_groups
    q, scale = quantise_groups(w, FORMATS["int8"], w.shape[-1], "exact",
                               torch.float32)
    return q.reshape(w.shape).to(torch.int8), scale.reshape(-1)


def _quant_matmul_build(inv: Invocation, env: Env,
                        inputs: Dict[str, Any]) -> BenchCase:
    import torch
    import torch_npu

    # Both operands arrive WIDE — int8 is too narrow to hold a reference
    # value (`refs._store_dtype`), so the cell carries bf16 and each
    # provider quantises, the convention the fp8 providers set.
    x, w = inputs["x"], inputs["w"]
    # The activation is quantised by the very kernel the plugin uses for it
    # (`w8a8_dynamic.py:117`), so its elements and per-token scale are that
    # kernel's, not a second opinion about what it writes.
    xq, per_token = torch_npu.npu_dynamic_quant(x, dst_type=torch.int8)
    wq, w_scale = _int8_per_channel(w)
    # The kernel takes the weight as (K, N): the plugin transposes once at
    # load (`w8a8_dynamic.py:135-139` process_weights_after_loading), which
    # is not per-step work and so is not inside the timed window.
    #
    # ALWAYS, not when the extents happen to differ. The cell states the
    # layout — `gemm` carries `w` as [N, K] and its oracle computes
    # `x @ w.T` — so sniffing the shapes answers a question the declaration
    # already answered, and answers it wrong whenever N == K: a square
    # weight went through untransposed and the kernel computed `x @ w`.
    # Nothing caught it because int8 had no cross-check band, so these cells
    # were measured with no oracle at all; with one, they read cosine
    # 0.00004 — orthogonal, not imprecise.
    wt = wq.t().contiguous()
    # The weight scale's width is the checkpoint's, which the cell carries
    # as `metadata` (BF16 for W8A8_DYNAMIC); the activation scale's is the
    # quantise kernel's own F32. Two facts, two widths.
    scale = w_scale.to(_torch_dtype(inv.dtype_env.get("metadata") or "bf16"))
    out_dtype = _torch_dtype(inv.dtype if inv.dtype in _ASCEND_FLOAT
                             else "bf16")
    return BenchCase(
        run=lambda: torch_npu.npu_quant_matmul(
            xq, wt, scale, pertoken_scale=per_token, output_dtype=out_dtype),
        # The kernel folds both scales into its epilogue, so its output is
        # already the wide product the oracle computes.
        to_canonical=lambda o: o,
    )


register(Provider(
    library="vllm_ascend", signature="gemm",
    kernel_impl="npu_quant_matmul_w8a8", import_name="torch_npu",
    build=_quant_matmul_build,
    serves=KernelSignature(signature="gemm", weight=("int8",),
                           activation=("int8",), output=_ASCEND_FLOAT,
                           # W8A8_DYNAMIC is per-channel weights and
                           # per-token activations, and the build above
                           # passes exactly those two scales — the same pair
                           # the plugin passes
                           # (`w8a8_dynamic.py:124-131`). Declared because a
                           # scale layout is part of what a kernel serves:
                           # undeclared, this provider served the plain cell
                           # only, so every W8A8 cell — the one deployment
                           # this binding exists for — found no provider.
                           weight_granularity=("per-channel",),
                           activation_granularity=("per-token",)),
))


def _grouped_quant_matmul_build(inv: Invocation, env: Env,
                                inputs: Dict[str, Any]) -> BenchCase:
    import torch
    import torch_npu

    x, w = inputs["x"], inputs["w"]
    counts = list(inputs["token_counts"])
    # `group_list` is CUMULATIVE token counts, one entry per expert
    # (`w8a8_dynamic.py:203-239` builds it the same way).
    cumulative, running = [], 0
    for count in counts:
        running += int(count)
        cumulative.append(running)
    groups = torch.tensor(cumulative, dtype=torch.int64, device=x.device)
    # The grouped kernel takes the tokens of every expert in one flat run,
    # not the [E, S, K] block the cell carries.
    # The cell's row block is [E, s_expert_rows, K] and its counts say how
    # many of each expert's rows are real. The kernel reads ONE flat run of
    # `running` tokens, so the padding between a count and its block's end
    # is dropped here rather than handed over: passed whole, the last
    # cumulative entry falls short of the tensor's length and the kernel
    # refuses the pair.
    rows = x.reshape(-1, x.shape[-1])
    per_block = rows.shape[0] // max(len(counts), 1)
    kept = [rows[i * per_block:i * per_block + int(c)]
            for i, c in enumerate(counts)]
    flat = (torch.cat(kept, dim=0).contiguous() if kept
            else rows[:running].contiguous())
    # As the dense provider: the operands arrive wide and this build
    # quantises them, the activation through the plugin's own quantise
    # kernel and the weight on the harness's grid, one scale per output
    # channel of each expert.
    flat, per_token = torch_npu.npu_dynamic_quant(flat, dst_type=torch.int8)
    wq, w_scale = _int8_per_channel(w.reshape(-1, w.shape[-1]))
    wq = wq.reshape(w.shape)
    w_scale = w_scale.reshape(w.shape[0], w.shape[1])
    # The cell carries each expert's weight as (N, K) — the layout a single
    # expert's linear states (`refs._grouped_gemm_inputs`) — while the
    # kernel contracts over the LAST axis and wants (K, N). Transposed once
    # here, outside the timed closure; the plugin transposes at load too
    # (`w8a8_dynamic.py:135-139`), so it is not per-step work either way.
    wt = wq.transpose(-1, -2).contiguous()
    # The two scales W8A8_DYNAMIC dequantises with, in the layout the plugin
    # hands the kernel (`device/device_op.py:418-426`): the weight's one per
    # output channel of each expert, the activation's one per token of the
    # flat run. Without them `npu_grouped_matmul` takes its NON-quantised
    # path (the op's own formula 1) — which is what this build did until
    # 2026-11-16, so a provider named `_w8a8` was timing the plain kernel on
    # int8 operands, and `to_canonical=None` meant nothing compared the
    # result against anything.
    # The WEIGHT scale is the checkpoint's, so its width is the cell's
    # `metadata` — BF16 for W8A8_DYNAMIC, which stores the sidecar at the
    # model dtype (`quant_schemes/modelslim-w8a8-dynamic.yaml`). Read from
    # the cell rather than fixed here: the kernel refuses the pair outright
    # otherwise ("scale[0] dtype FLOAT and y dtype BF16 are not supported
    # when A8W8 per-token quant", aclnnGroupedMatmulV5 161002), so a
    # hardcoded width would be a second opinion about a fact the cell
    # already carries.
    scale = w_scale.to(_torch_dtype(inv.dtype_env.get("metadata") or "bf16"))
    out_dtype = _torch_dtype(inv.dtype if inv.dtype in _ASCEND_FLOAT
                             else "bf16")
    return BenchCase(
        run=lambda: torch_npu.npu_grouped_matmul(
            [flat], [wt], scale=[scale], per_token_scale=[per_token],
            group_list=groups, split_item=2, group_type=0,
            output_dtype=out_dtype),
        to_canonical=None,   # flat-vs-blocked layout; measured, not verified
    )


register(Provider(
    library="vllm_ascend", signature="grouped_gemm",
    kernel_impl="npu_grouped_matmul_w8a8", import_name="torch_npu",
    build=_grouped_quant_matmul_build,
    serves=KernelSignature(signature="grouped_gemm", weight=("int8",),
                           activation=("int8",), output=_ASCEND_FLOAT,
                           # As the dense provider above, and for the same
                           # reason: the scales are what makes this the W8A8
                           # kernel rather than the float one.
                           weight_granularity=("per-channel",),
                           activation_granularity=("per-token",)),
))


def _grouped_matmul_build(inv: Invocation, env: Env,
                          inputs: Dict[str, Any]) -> BenchCase:
    import torch
    import torch_npu

    x, w = inputs["x"], inputs["w"]
    counts = list(inputs["token_counts"])
    cumulative, running = [], 0
    for count in counts:
        running += int(count)
        cumulative.append(running)
    groups = torch.tensor(cumulative, dtype=torch.int64, device=x.device)
    # The cell's row block is [E, s_expert_rows, K] and its counts say how
    # many of each expert's rows are real. The kernel reads ONE flat run of
    # `running` tokens, so the padding between a count and its block's end
    # is dropped here rather than handed over: passed whole, the last
    # cumulative entry falls short of the tensor's length and the kernel
    # refuses the pair.
    rows = x.reshape(-1, x.shape[-1])
    per_block = rows.shape[0] // max(len(counts), 1)
    kept = [rows[i * per_block:i * per_block + int(c)]
            for i, c in enumerate(counts)]
    flat = (torch.cat(kept, dim=0).contiguous() if kept
            else rows[:running].contiguous())
    # The cell carries each expert's weight as (N, K) — the layout a single
    # expert's linear states (`refs._grouped_gemm_inputs`) — while the
    # kernel contracts over the LAST axis and wants (K, N). Transposed once
    # here, outside the timed closure; the plugin transposes at load too
    # (`w8a8_dynamic.py:135-139`), so it is not per-step work either way.
    wt = w.transpose(-1, -2).contiguous()
    return BenchCase(
        run=lambda: torch_npu.npu_grouped_matmul(
            [flat], [wt], group_list=groups, split_item=2, group_type=0),
        to_canonical=None,
    )


register(Provider(
    library="vllm_ascend", signature="grouped_gemm",
    kernel_impl="npu_grouped_matmul", import_name="torch_npu",
    build=_grouped_matmul_build,
    serves=KernelSignature(signature="grouped_gemm", weight=_ASCEND_FLOAT,
                           activation=_ASCEND_FLOAT, output=_ASCEND_FLOAT),
))


def _dynamic_quant_build(inv: Invocation, env: Env,
                         inputs: Dict[str, Any]) -> BenchCase:
    import torch
    import torch_npu

    x = inputs["x"]
    cast = inputs["cast"]

    def canonical(pair):
        """The kernel's `(elements, scale)` multiplied back.

        The oracle is `x` quantised and dequantised again
        (`ActivationCast.on_grid`), i.e. a WIDE tensor; the kernel writes the
        narrow pair. Handing the pair over untouched compared int8 element
        VALUES against dequantised activations — same direction, ~127x the
        magnitude, which is what the cross-check read as cosine 0.997 with a
        typical error of 15.5.

        `npu_dynamic_quant` returns one scale per token, shaped like the
        activation with its last axis dropped; `quantise_groups` states the
        canonical shape as `[rows, groups]`, and per-token is one group per
        row. So the scale is reshaped, and the multiply itself is the cast's
        own `dequantise` — the oracle's too, so the two cannot drift into
        different scale conventions.
        """
        elements, scale = pair[0], pair[1]
        return cast.dequantise((elements, scale.reshape(-1, 1)))

    return BenchCase(
        run=lambda: torch_npu.npu_dynamic_quant(x, dst_type=torch.int8),
        to_canonical=canonical,
    )


register(Provider(
    library="vllm_ascend", signature="quantise_activation",
    kernel_impl="npu_dynamic_quant_int8", import_name="torch_npu",
    build=_dynamic_quant_build,
    serves=KernelSignature(signature="quantise_activation",
                           activation=_ASCEND_FLOAT, output=("int8",),
                           activation_granularity=("per-token",),
                           # One FP32 scale per token — measured on a die,
                           # not read off a doc: `npu_dynamic_quant` of a
                           # [128, 512] bf16 activation returns int8
                           # elements and a float32 scale of shape [128].
                           # This is the width `int8_per_token.yaml`
                           # declares, so the two now agree on evidence.
                           dtype_slots=(("metadata", ("fp32",)),)),
))


# ---------------------------------------------------------------------------
# The hyper-connection seam, as vLLM-Ascend runs it for GLM-5.3-Flash.
#
# Both halves live in the plugin's own compiled extension rather than in
# `torch_npu`, so reaching them needs two things that no other provider
# here needs, and both are done once in `_c_ascend()`:
#
#   * the extension module has to be imported — `torch.ops._C_ascend` is an
#     empty namespace until `vllm_ascend.vllm_ascend_C` is loaded;
#   * CANN has to be told where the plugin's custom operator package is.
#     The plugin prepends it to `ASCEND_CUSTOM_OPP_PATH` itself
#     (`vllm_ascend/utils.py:455-458`); a bare process that skips that
#     prepend gets the op registered and then "aclnnHcPre ... not in
#     libopapi.so" when it runs, which is a loud failure rather than a
#     wrong number, but it is not the kernel declining either.
#
# `torch_npu.npu_mhc_pre` / `npu_mhc_post` exist and take the same maths,
# and are NOT used: on this CANN and this chip the first of them
# segfaults the process. The plugin calls the `_C_ascend` pair, which is
# also what the profiler shows (`HcPre` / `HcPost`), so that is what gets
# measured.
# ---------------------------------------------------------------------------

#: The plugin's custom operator package, relative to its own package dir.
#: `vllm_ascend/utils.py:455-458` prepends exactly this to
#: `ASCEND_CUSTOM_OPP_PATH`, and the plugin does it at import time —
#: before anything touches a device.
_CUSTOM_OPP_REL = ("_cann_ops_custom", "vendors", "custom_transformer")

#: `ASCEND_CUSTOM_OPP_PATH` as this process found it, read at import of this
#: module — which is before any provider runs and before the first device
#: call. That is the only value that says anything: importing
#: `vllm_ascend` prepends the plugin's own package to the variable, so a
#: check made later always finds it and CANN has already read the variable
#: it had at initialisation.
_OPP_AT_IMPORT = tuple(
    p for p in __import__("os").environ.get(
        "ASCEND_CUSTOM_OPP_PATH", "").split(__import__("os").pathsep) if p)


def _c_ascend():
    """`torch.ops._C_ascend`, with the plugin's extension loaded.

    Two things have to be true before either hc op will run, and only one
    of them can be arranged from here:

      * the extension module must be imported — `torch.ops._C_ascend` is an
        empty namespace until `vllm_ascend.vllm_ascend_C` is loaded. Done
        below;
      * CANN must have the plugin's operator package on its search path,
        and it reads `ASCEND_CUSTOM_OPP_PATH` **once, at initialisation**.
        Setting it from a provider that runs after the first device call is
        too late: the op registers, then fails at run time with "The tiling
        function does not exist" (EZ1008) — measured, not inferred. So this
        CHECKS the variable and refuses, rather than setting it and
        returning a launch that cannot run. The box profile exports it
        (`kernel_bench/envs/boxes/ascend-910c-a3-cann82.toml`), which is
        where a process-wide environment fact belongs.
    """
    import os

    import torch
    import vllm_ascend

    vendor = os.path.join(os.path.dirname(vllm_ascend.__file__),
                          *_CUSTOM_OPP_REL)
    # Judged against what the PROCESS inherited, not what is there now:
    # the `import vllm_ascend` two lines up added `vendor` to the live
    # variable whether or not CANN ever saw it.
    if not any(p.rstrip(os.sep) == vendor.rstrip(os.sep)
               for p in _OPP_AT_IMPORT):
        raise NotImplementedError(
            f"the plugin's custom operator package is not on "
            f"ASCEND_CUSTOM_OPP_PATH, and CANN reads that variable once at "
            f"initialisation, so this process cannot add it: export "
            f"ASCEND_CUSTOM_OPP_PATH={vendor} before starting the run "
            f"(the box profile does)")
    import vllm_ascend.vllm_ascend_C  # noqa: F401  (registers the ops)
    return torch.ops._C_ascend


def _mhc_pre_build(inv: Invocation, env: Env,
                   inputs: Dict[str, Any]) -> BenchCase:
    """`npu_hc_pre_v2`: the whole pre-mix, one launch.

    The kernel takes `[T, n, D]` — one token per row, never a batch dim,
    which is the layout the plugin hands it (`model.py:425`: the model's
    stream is `[tokens, n, d]` throughout). The signature writes the axes
    as `[B, s_query, n, d]`, which the walk folds into `[tokens, n, d]`
    whenever B is 1, so both ranks reach this build; whatever is not the
    last two axes is the token axis.

    The three scalars that are not axes ride on the inputs, where every
    consumer of this span reads them (`refs._hc_scalars`); the Sinkhorn
    round count is an axis and comes from `dims`, as the scan chunk does
    for `mamba2_scan`.
    """
    import torch

    ops = _c_ascend()
    residual = inputs["residual"]
    n, d = residual.shape[-2], residual.shape[-1]
    x = residual.reshape(-1, n, d).contiguous()
    fn = inputs["fn"].float().contiguous()
    scale = inputs["scale"].float().contiguous()
    base = inputs["base"].float().contiguous()
    iters = int(inv.dims["sinkhorn_repeat"])
    rms_eps = float(inputs["rms_eps"])
    hc_eps = float(inputs["hc_pre_eps"])

    def run():
        return ops.npu_hc_pre_v2(x, fn, scale, base, n, iters,
                                 rms_eps, hc_eps)

    # The launch writes three tensors and `_cross_check` compares one, so
    # they are folded side by side exactly as the oracle folds them
    # (`refs._hc_pre_reference`). `post` arrives `[T, n]` and `comb`
    # `[T, n, n]`, both fp32.
    return BenchCase(
        run=run,
        to_canonical=lambda o: torch.cat(
            [o[0].float(), o[1].float(), o[2].float().flatten(1)], dim=-1),
    )


register(Provider(
    library="vllm_ascend", signature="mhc_pre",
    kernel_impl="npu_hc_pre_v2", import_name="vllm_ascend",
    build=_mhc_pre_build, fused_op="mhc_pre",
    serves=KernelSignature(signature="mhc_pre", activation=_ASCEND_FLOAT),
))


def _mhc_post_build(inv: Invocation, env: Env,
                    inputs: Dict[str, Any]) -> BenchCase:
    """`npu_hc_post`: the closing post-mix, one launch.

    The plugin calls it with a leading singleton on every operand and
    squeezes it off the result (`model.py:523-525`), so the cell's
    `[B, s_query, ...]` tensors fold to `[T, ...]` and take that singleton
    back — the same tensors, in the layout the kernel was registered for.
    `post` reaches the cell `[T, n]`; the plugin carries a trailing 1 on it
    between the two halves and drops it here, so the call takes it flat.
    """
    ops = _c_ascend()
    sublayer, residual = inputs["sublayer"], inputs["residual"]
    post, comb = inputs["post"], inputs["comb"]
    # Everything but the last (one for the sublayer, two for the residual
    # and the cross-mix) axes is the token axis, as in the pre-mix; the
    # plugin's leading singleton is put back here because that is the rank
    # the op was registered for.
    x = sublayer.reshape(1, -1, sublayer.shape[-1]).contiguous()
    res = residual.reshape(1, -1, residual.shape[-2],
                           residual.shape[-1]).contiguous()
    gate = post.reshape(1, -1, post.shape[-1]).float().contiguous()
    mix = comb.reshape(1, -1, comb.shape[-2],
                       comb.shape[-1]).float().contiguous()

    def run():
        return ops.npu_hc_post(x, res, gate, mix)

    return BenchCase(
        run=run,
        to_canonical=lambda o: o.squeeze(0).float(),
    )


register(Provider(
    library="vllm_ascend", signature="mhc_post",
    kernel_impl="npu_hc_post", import_name="vllm_ascend",
    build=_mhc_post_build, fused_op="mhc_post",
    serves=KernelSignature(signature="mhc_post", activation=_ASCEND_FLOAT),
))


# ---------------------------------------------------------------------------
# Per-head matmuls and the delta-rule linear attention.
#
# `npu_transpose_batchmatmul` is what the SFA backend runs MLA's absorbed
# `W_UK` / `W_UV` through (vllm_ascend/attention/sfa_v1.py:1342 and :1373),
# and the profiler shows 264 of them in a 12-step trace — 12 steps x 11 MLA
# layers x 2, none in a KDA layer. The fallback those call sites keep for
# old builds is `torch.bmm` on a transposed operand; measured against it on
# one die this call agrees at cosine 0.9999988.
#
# `chunk_kda_fwd` / `recurrent_kda` are the two halves of KDA, and which one
# runs is the PHASE: the plugin reads the cache through
# `fla_npu.ops.ascendc`, chunked in prefill and recurrent in decode
# (vllm_ascend/models/glm5next/ops/kda.py:16-30 and :60-110). Two providers
# on one signature, separated by `phases` — the same arrangement vLLM's own
# `kda` pair and the Ascend `gqa_attention` pair already use.
# ---------------------------------------------------------------------------

def _tbm_per_head_build(inv: Invocation, env: Env,
                        inputs: Dict[str, Any]) -> BenchCase:
    """`npu_transpose_batchmatmul` over `[B, H, S, K] x [H, N, K]`.

    The operator takes its activations token-major and permutes internally,
    which is why the call passes `perm_x1=(1, 0, 2)`: `[T, H, K]` is read as
    `[H, T, K]`, the batch dim being the head. `perm_y=(1, 0, 2)` puts the
    result back token-major. The plugin's call site is handed the tensor in
    exactly that form, so the reshape below is the layout the kernel was
    given, not a conversion this build invents.
    """
    import torch
    import torch_npu

    x, w = inputs["x"], inputs["w"]          # [B,H,S,K], [H,N,K]
    B, H, S, K = x.shape
    N = w.shape[1]
    # The op rejects a batch operand dim at or past 65536
    # (`TRANSPOSE_BMM_MAX_SUPPORTED_DIM`, sfa_v1.py:329), and the internal
    # transpose needs H*K under it too (sfa_v1.py:1366-1372). A cell past
    # either is the fallback's launch, not this one's.
    if B * S >= 65536 or H * K >= 65536:
        raise NotImplementedError(
            f"npu_transpose_batchmatmul takes operand dims under 65536; this "
            f"cell is tokens={B * S}, n_heads*d_reduce={H * K}")

    x_t = x.permute(0, 2, 1, 3).reshape(-1, H, K).contiguous()
    w_t = w.transpose(1, 2).contiguous()     # [H,N,K] -> [H,K,N]

    def run():
        return torch_npu.npu_transpose_batchmatmul(
            x_t, w_t, bias=None, scale=None,
            perm_x1=(1, 0, 2), perm_x2=(0, 1, 2), perm_y=(1, 0, 2),
            batch_split_factor=1)

    # [T,H,N] back to the canonical [B,H,S,N] the oracle produces.
    return BenchCase(
        run=run,
        to_canonical=lambda o: o.reshape(B, S, H, N).permute(0, 2, 1, 3))


register(Provider(
    library="vllm_ascend", signature="per_head_gemm",
    kernel_impl="npu_transpose_batchmatmul", import_name="torch_npu",
    build=_tbm_per_head_build,
    serves=KernelSignature(signature="per_head_gemm",
                           activation=_ASCEND_FLOAT, weight=_ASCEND_FLOAT,
                           output=_ASCEND_FLOAT),
))


def _kda_bsnd(inputs: Dict[str, Any]):
    """The cell's operands in the layout the AscendC kernels take.

    The cell states `q` as `[B, s_query, n_heads, d]` — BSND, which is what
    `layout="BSND"` means to `fla_npu` — while the ENGINE layout in the
    catalog is `[B, n_heads, s_query, d]`, the shape the framework emits.
    A provider is handed the CANONICAL contract, so nothing is transposed
    here: the kernels' own layout argument says which one they want.

    The BATCH goes too, and that is the one thing this build does reshape.
    Both entry points are varlen: `cu_seqlens` says where each sequence
    starts, and the plugin hands them `[1, tokens, heads, d]` with one
    leading dim whatever the batch (`models/glm5next/kda.py:534-570`
    reshapes `q_ns` to `[1, -1, h, d]`). A cell stating `[B, s, h, d]` is
    the same tokens with the batch still split out, so it folds here and
    `cu_seqlens` / the state slots carry `B` instead — which is the shape
    the operator accepts: `[B, s, …]` is refused with ACLNN 161002.
    """
    import torch
    B, S = inputs["q"].shape[0], inputs["q"].shape[1]
    q, k, v = (t.reshape(1, B * S, t.shape[2], t.shape[3]).contiguous()
               for t in (inputs["q"], inputs["k"], inputs["v"]))
    g = inputs["g"].reshape(1, B * S, inputs["g"].shape[2],
                            inputs["g"].shape[3]).float().contiguous()
    beta = inputs["beta"].reshape(1, B * S, inputs["beta"].shape[2]).contiguous()
    # One sequence per batch row, laid end to end.
    cu = [i * S for i in range(B + 1)]
    return q, k, v, g, beta, cu, B, S


def _chunk_kda_build(inv: Invocation, env: Env,
                     inputs: Dict[str, Any]) -> BenchCase:
    """`chunk_kda_fwd`: the prefill half, one launch over the whole block.

    The gate arrives ALREADY FORMED (`g` is the per-channel log decay the
    walk's gate chain produced) and `beta` already activated, so this is the
    span with `use_gate_in_kernel=False` — not the fused-gate variant the
    plugin's own call site runs, which is a different span
    (`fused_kda_gate`) and a different provider. The q/k norm IS inside, as
    the oracle applies it (`refs._delta_rule_scan` normalises both), and as
    the device's recurrent half takes it.

    The chunk size is the kernel's own 64 (constant in the plugin's
    `run_chunk_kda`), not an axis: it does not divide the query block in a
    way the cell's identity distinguishes, and the plugin passes the
    constant.
    """
    from fla_npu.ops.ascendc import chunk_kda_fwd

    q, k, v, g, beta, cu, B, S = _kda_bsnd(inputs)
    state = inputs["initial_state"].float().contiguous()
    scale = float(q.shape[-1]) ** -0.5

    def run():
        return chunk_kda_fwd(
            q, k, v, g, beta, scale, 64,
            layout="BSND", initial_state=state, output_final_state=True,
            cu_seqlens=cu, safe_gate=False, use_qk_l2norm_in_kernel=True,
            return_intermediate_states=False, state_v_first=True)

    def canonical(out):
        """`(o, final_state, ...)` in the kernel's varlen form back to the
        cell's `[B, s_query, n_heads, d]`.

        The launch writes `[1, B·s_query, h, d]` — one leading dim, the
        batch folded into the tokens — so the result is split back along
        that token axis at `s_query`. Skipping this handed the cross-check a
        `[1, B·S, h, d]` tensor against a `[B, S, h, d]` reference: the same
        numbers in a different shape, which the gate reports as a shape
        mismatch and the row is never written.
        """
        o = out[0] if isinstance(out, (tuple, list)) else out
        return o.reshape(B, S, *o.shape[2:]).float()

    return BenchCase(run=run, to_canonical=canonical)


register(Provider(
    library="vllm_ascend", signature="kda",
    kernel_impl="npu_chunk_kda_fwd", import_name="fla_npu",
    build=_chunk_kda_build,
    serves=KernelSignature(signature="kda", phases=("prefill",),
                           activation=_ASCEND_FLOAT, takes_state=True),
))


#: Tokens per sequence the recurrent step takes on the path this provider
#: models: exactly one. The plugin's own constant is 8
#: (`models/glm5next/kda.py:151`, `ops/kda.py:11`), but that is the bound on
#: a SPECULATIVE step — `num_spec + 1` tokens per sequence, whose
#: `cu_seqlens` are built from the accepted-token counts and whose state
#: slots advance per accepted token. Measured on a die, the plain path takes
#: one token per sequence and refuses anything else with ACLNN 161002
#: whether the axis is stated as `s_query` or as a longer `cu_seqlens`
#: (B=16 with s_query 1 runs at 68.5 us; the same cell at 4 raises). Rather
#: than model the speculative variant's bookkeeping, this declines above
#: one, so a cell it cannot launch is never measured as if it could.
_KDA_RECURRENT_TOKENS = 1


def _recurrent_kda_build(inv: Invocation, env: Env,
                         inputs: Dict[str, Any]) -> BenchCase:
    """`recurrent_kda`: the decode half, one launch for the step.

    Two shape facts this build settles, both read off the plugin's own call
    site rather than guessed:

      * the state is a POOL plus one slot index per sequence, not a dense
        `[N, H, V, K]` — `run_recurrent_kda` hands the cache pool and
        `ssm_state_indices` through (ops/kda.py:16-30), which is sglang's
        shape of this call and not vLLM's chunked one (`_initial_state_args`
        records that split). The cell's state is one sequence's worth, so
        the pool is `[B+1, H, V, K]` with slot 0 unused and `B` slots used —
        exactly what the GPU provider for this span builds.
      * `state_v_first=True`: the plugin's state is `[V, K]`, the cell's
        `[d_lin_v_head, d_lin_k_head]`.
    """
    import torch
    from fla_npu.ops.ascendc import recurrent_kda

    q, k, v, g, beta, cu_list, B, S = _kda_bsnd(inputs)
    if S != _KDA_RECURRENT_TOKENS:
        raise NotImplementedError(
            f"the recurrent step takes {_KDA_RECURRENT_TOKENS} token per "
            f"sequence and this cell has s_query={S}. A whole block is the "
            f"chunked kernel's — the prefill provider on this signature. "
            f"The plugin's own 8-token bound is its SPECULATIVE step, whose "
            f"cu_seqlens and state slots follow the accepted-token counts "
            f"and are not this launch")
    _one, _tokens, HV, V = v.shape
    K = q.shape[-1]
    dense = inputs["initial_state"].float()
    pool = torch.zeros(B + 1, HV, V, K, dtype=torch.float32,
                       device=dense.device)
    pool[1:] = dense
    slots = torch.arange(1, B + 1, dtype=torch.int32, device=dense.device)
    cu = torch.tensor(cu_list, dtype=torch.int32, device=dense.device)
    scale = float(K) ** -0.5

    def run():
        return recurrent_kda(
            q, k, v, g, beta, pool, cu_seqlens=cu,
            ssm_state_indices=slots, layout="BSND", scale=scale,
            output_final_state=False, inplace_final_state=True,
            state_v_first=True, use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False, safe_gate=False)

    def canonical(out):
        """`[1, B, h, d]` back to the cell's `[B, 1, h, d]`: one token per
        sequence, split by the batch (`_kda_bsnd` folds them, this unfolds).
        A shape the cross-check reads, not a detail — see the chunk build."""
        o = out[0] if isinstance(out, (tuple, list)) else out
        return o.transpose(0, 1).float()

    return BenchCase(run=run, to_canonical=canonical)


register(Provider(
    library="vllm_ascend", signature="kda",
    kernel_impl="npu_recurrent_kda", import_name="fla_npu",
    build=_recurrent_kda_build,
    serves=KernelSignature(signature="kda", phases=("decode",),
                           activation=_ASCEND_FLOAT, takes_state=True),
))


# ---------------------------------------------------------------------------
# BF16 primitives and the bound fusions.
# ---------------------------------------------------------------------------

def _add_rmsnorm_build(inv: Invocation, env: Env,
                       inputs: Dict[str, Any]) -> BenchCase:
    import torch_npu

    x, residual, weight = inputs["x"], inputs["residual"], inputs["w"]
    eps = float(getattr(inv, "eps", 1e-6) or 1e-6)
    return BenchCase(
        run=lambda: torch_npu.npu_add_rms_norm(x, residual, weight, eps),
        to_canonical=lambda o: o[0] if isinstance(o, (tuple, list)) else o,
    )


register(Provider(
    library="vllm_ascend", signature="fused_add_rmsnorm",
    kernel_impl="npu_add_rms_norm", import_name="torch_npu",
    build=_add_rmsnorm_build, fused_op="fused_add_rmsnorm",
    serves=KernelSignature(signature="fused_add_rmsnorm",
                           activation=_ASCEND_FLOAT),
))


def _swiglu_build(inv: Invocation, env: Env,
                  inputs: Dict[str, Any]) -> BenchCase:
    import torch_npu

    x = inputs["x"]
    return BenchCase(run=lambda: torch_npu.npu_swiglu(x),
                     to_canonical=lambda o: o)


register(Provider(
    library="vllm_ascend", signature="silu_and_mul",
    kernel_impl="npu_swiglu", import_name="torch_npu",
    build=_swiglu_build, fused_op="silu_and_mul",
    serves=KernelSignature(signature="silu_and_mul",
                           activation=_ASCEND_FLOAT),
))


def _rmsnorm_build(inv: Invocation, env: Env,
                   inputs: Dict[str, Any]) -> BenchCase:
    import torch_npu

    x, weight = inputs["x"], inputs["w"]
    eps = float(getattr(inv, "eps", 1e-6) or 1e-6)
    return BenchCase(
        run=lambda: torch_npu.npu_rms_norm(x, weight, epsilon=eps),
        to_canonical=lambda o: o[0] if isinstance(o, (tuple, list)) else o,
    )


register(Provider(
    library="vllm_ascend", signature="rmsnorm",
    kernel_impl="npu_rms_norm", import_name="torch_npu",
    build=_rmsnorm_build,
    serves=KernelSignature(signature="rmsnorm", activation=_ASCEND_FLOAT),
))


# ---------------------------------------------------------------------------
# Attention: one `npu_fused_infer_attention_score` launch, prefill and decode
# (`vllm_ascend/attention/attention_v1.py:875-1019`; MLA
# `attention/mla_v1.py:1749-1751` prefill, `:3051` decode).
#
# THE PAGE, AND WHY THESE CELLS ARE NOT THE CUDA ONES
#
# The decode build reads the cache through a page table at the page the
# plugin allocates (128). CANN requires the paged layout to be BNSD with the
# cache as `[blocks, n_kv_heads, page, d_head]`: in BSND it reads the page
# axis as N and refuses ("When page attention is enabled, the axis N of key
# must be equal to N2"), which is how the layout below was settled — on the
# device, not from the signature's docstring.
#
# Prefill takes the contiguous KV the cell carries and pages nothing, so the
# two builds declare different `kv_block_sizes` and are different cells.
# ---------------------------------------------------------------------------

def _fia_prefill_build(inv: Invocation, env: Env,
                       inputs: Dict[str, Any]) -> BenchCase:
    import torch
    import torch_npu

    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    heads = int(inv.dims["n_heads"])
    kv_heads = int(inv.dims.get("n_kv_heads", heads))
    scale = float(q.shape[-1]) ** -0.5
    causal = bool(inputs.get("causal", True))

    # CAUSAL PREFILL IS DECLINED, and the reason is the kernel's interface
    # rather than this build's shape handling.
    #
    # FIA applies a causal mask only when handed a FULL [s_query, s_kv] bool
    # tensor. Measured: `pre_tokens`/`next_tokens` read like a causal window
    # and change nothing at all (with them, without them and at
    # `sparse_mode=0` the output was identical, and all three matched the
    # NON-causal reference at cosine 1.0 while missing the causal one at
    # 0.41); every `sparse_mode` without a mask tensor raises; a mask
    # smaller than the cell raises. So the only correct call carries a
    # bool[s_query, s_kv].
    #
    # That tensor is larger than the cell's own operands from s_query 2048
    # up — 4M bool against 1M bf16 of q — and the harness refuses a launch
    # whose widest buffer is not the cell's precision
    # (`launch_trace`, "its latency is another precision's"). The refusal is
    # right: those mask bytes are real traffic this cell does not declare,
    # so a row measured here would price a mask the model never stores.
    #
    # Declined rather than measured wrong, and declined rather than
    # silently measured non-causal, which is the dangerous third option: it
    # returns a number for twice the work the cell asks for. Fixing it means
    # the cell declaring the mask operand — a signature change, not a build
    # one.
    if causal:
        raise NotImplementedError(
            "FIA takes a causal prefill only with a full bool[s_query, s_kv] "
            "mask, which outweighs the cell's own operands and makes the "
            "launch another precision's; the cell would have to declare that "
            "operand")

    def run():
        out = torch_npu.npu_fused_infer_attention_score(
            q, k, v, num_heads=heads, num_key_value_heads=kv_heads,
            input_layout="BSND", scale=scale)
        return out[0] if isinstance(out, (list, tuple)) else out

    return BenchCase(run=run, to_canonical=lambda o: o)


register(Provider(
    library="vllm_ascend", signature="gqa_attention",
    kernel_impl="npu_fused_infer_attention_score", import_name="torch_npu",
    build=_fia_prefill_build, fused_op="flash_attention",
    serves=KernelSignature(signature="gqa_attention", phases=("prefill",),
                           activation=_ASCEND_FLOAT,
                           # FIA reads the cache at the activation's own
                           # width: the plugin hands it the KV it stored,
                           # and a quantised cache is a different kernel
                           # (the C8 path), not this one.
                           dtype_slots=(("kv", _ASCEND_FLOAT),),
                           # Contiguous KV, as the cell carries it: this
                           # build allocates no page table.
                           kv_block_sizes=()),
))


def _fia_decode_build(inv: Invocation, env: Env,
                      inputs: Dict[str, Any]) -> BenchCase:
    import torch
    import torch_npu

    q, k, v = inputs["q"], inputs["k"], inputs["v"]
    heads = int(inv.dims["n_heads"])
    kv_heads = int(inv.dims.get("n_kv_heads", heads))
    batch = int(inv.dims["B"])
    kv_len = int(inv.dims["s_kv"])
    d_head = int(q.shape[-1])
    scale = float(d_head) ** -0.5

    # The cell carries contiguous KV; a paged kernel reads the same bytes
    # through a page table, so the pages are a VIEW of that cache rather
    # than a second allocation — the bytes measured stay the cell's.
    pages_per_seq = (kv_len + ASCEND_PAGE - 1) // ASCEND_PAGE
    blocks = batch * pages_per_seq
    padded = blocks * ASCEND_PAGE

    def _as_pages(t):
        # [B, S, N, D] -> [blocks, N, page, D] (BNSD, which is the layout
        # CANN requires once paging is on).
        flat = t.reshape(batch, -1, kv_heads, d_head)
        if flat.shape[1] < padded // batch:
            pad = torch.zeros(batch, padded // batch - flat.shape[1],
                              kv_heads, d_head, dtype=t.dtype,
                              device=t.device)
            flat = torch.cat([flat, pad], dim=1)
        return (flat.reshape(blocks, ASCEND_PAGE, kv_heads, d_head)
                .permute(0, 2, 1, 3).contiguous())

    k_pages, v_pages = _as_pages(k), _as_pages(v)
    table = (torch.arange(blocks, dtype=torch.int32, device=q.device)
             .reshape(batch, pages_per_seq))
    # BNSD for the query too: [B, N, S, D].
    q_bnsd = q.reshape(batch, -1, heads, d_head).permute(0, 2, 1, 3).contiguous()
    seq_lengths = [kv_len] * batch

    def run():
        out = torch_npu.npu_fused_infer_attention_score(
            q_bnsd, k_pages, v_pages,
            num_heads=heads, num_key_value_heads=kv_heads,
            input_layout="BNSD", scale=scale,
            block_table=table, block_size=ASCEND_PAGE,
            actual_seq_lengths_kv=seq_lengths)
        return out[0] if isinstance(out, (list, tuple)) else out

    # Back to the canonical [B, S, N, D] the signature's reference produces.
    return BenchCase(run=run,
                     to_canonical=lambda o: o.permute(0, 2, 1, 3).contiguous())


register(Provider(
    library="vllm_ascend", signature="gqa_attention",
    kernel_impl="npu_fused_infer_attention_score_paged",
    import_name="torch_npu",
    build=_fia_decode_build, fused_op="flash_attention",
    serves=KernelSignature(signature="gqa_attention", phases=("decode",),
                           activation=_ASCEND_FLOAT,
                           dtype_slots=(("kv", _ASCEND_FLOAT),),
                           # The page vLLM-Ascend allocates the cache in.
                           # Disjoint from every CUDA backend's 16/32/64, so
                           # these are their own cells — which is the fact,
                           # not an inconvenience: a kernel reading 128-token
                           # pages gathers a quarter as many pages per query.
                           kv_block_sizes=(ASCEND_PAGE,)),
))


# ---------------------------------------------------------------------------
# HCCL collectives.
#
# The four signatures have existed since multi-GPU totals were wanted
# (`op_signatures_catalog.py:1367-1420`: "without these, a multi-GPU
# deployment could never have a measured total") and have never had a
# provider. These are the first.
#
# REGISTERED AND DISABLED, which is a declaration rather than a gap. The
# harness runs ONE process (`driver.py` has no rank, no world size), and in
# one process every one of these collectives is the identity: the call
# returns, the clock reads a local copy, and the row says an N-rank
# all-reduce costs what a memcpy costs. That is exactly how the one
# collective anyone had measured — `allreduce_fused_rmsnorm` — came to be
# disabled with the same reason, and its note is the precedent followed
# here.
#
# What it would take, so the next reader does not re-derive it: a runner
# that starts `n_comm_ranks` processes, each on its own die, with the group
# initialised on HCCL — `torchrun --nproc_per_node=N`, `backend="hccl"`,
# `torch.npu.set_device(local_rank)`. The measurement itself is settled;
# what is missing is the harness's process model, not the kernel.
#
# MEASURED ANYWAY, out of band, because the sheet's link rate needed a
# bound (2026-11-15, one process per die, 128 MiB payloads, bus bandwidth
# through `comm_volume.ring_factor`):
#
#     16 dies   all-gather 186.4 GB/s, all-reduce 176.7
#      8 dies   132.8 GB/s pure HCCS (one die per package)
#
# against the 196 GB/s the sheet states. Recorded as a floor under the cell
# in `tests/backend/inference/test_ascend_peaks_bound_measurement.py`.
# ---------------------------------------------------------------------------

_SINGLE_PROCESS = (
    "one process: every collective is the identity, so this would time a "
    "local buffer and record it as an N-rank collective — a group of "
    "n_comm_ranks needs that many processes, each on its own die, with the "
    "group on HCCL")


def _hccl_build(op: str):
    """The build for one collective. `torch.distributed` is the entry point
    the plugin's own ranks use; the group is the process group HCCL
    initialised, which a single-process harness does not have — hence the
    disabled_reason above."""

    def build(inv: Invocation, env: Env, inputs: Dict[str, Any]) -> BenchCase:
        import torch.distributed as dist

        payload = inputs["payload"]
        if op == "all_reduce":
            run = lambda: dist.all_reduce(payload)
        elif op == "all_gather":
            import torch
            ranks = int(inputs["n_comm_ranks"])
            out = torch.empty(payload.numel() * ranks, dtype=payload.dtype,
                              device=payload.device)
            run = lambda: dist.all_gather_into_tensor(out, payload)
        elif op == "reduce_scatter":
            import torch
            ranks = int(inputs["n_comm_ranks"])
            out = torch.empty(payload.numel() // ranks, dtype=payload.dtype,
                              device=payload.device)
            run = lambda: dist.reduce_scatter_tensor(out, payload)
        elif op == "all_to_all":
            import torch
            out = torch.empty_like(payload)
            run = lambda: dist.all_to_all_single(out, payload)
        else:
            raise ValueError(op)
        # No oracle: on one process there is nothing to compare against, and
        # with a real group each rank holds a different answer.
        return BenchCase(run=run, to_canonical=None)

    return build


for _op in ("all_reduce", "all_gather", "reduce_scatter", "all_to_all"):
    register(Provider(
        library="vllm_ascend", signature=_op,
        kernel_impl=f"hccl_{_op}", import_name="torch_npu",
        build=_hccl_build(_op),
        serves=KernelSignature(signature=_op, activation=_ASCEND_FLOAT),
        disabled_reason=_SINGLE_PROCESS,
    ))
