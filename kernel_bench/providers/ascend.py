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

#: The paged-KV page vLLM-Ascend allocates the cache in, from the binding
#: (`framework_bindings/vllm_ascend.yaml` `kv_block_size`, read at
#: `vllm_ascend/utils.py:1504-1505`). Stated once here; a second literal in
#: a build would be a second statement of the framework's own choice.
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
