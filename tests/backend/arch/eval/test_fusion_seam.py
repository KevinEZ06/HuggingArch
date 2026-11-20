"""A launch across the seam between two layers (`eval.fusion.Seams`).

vLLM and sglang never finish a layer with its residual add: the next
layer's input norm takes it in the same launch (`fused_add_rmsnorm`, with
its quantised output where a quantised GEMM reads it), and the model's
final norm takes the last one. The matcher follows a template's formal
into the iteration that wrote it, so the plan matches across the seam, and
the rewrite takes a member's runs one seam at a time (`Rewrite.shares`):
the norm that opens a stack keeps its layer-0 run.

The counts and bytes are checked against a second route: the checkpoint's
own `config.json` (layer count, first dense layers, hidden width, FP8
block), not the walk's dims.
"""
from __future__ import annotations

import functools
from itertools import pairwise
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.eval.launch import FusedNode
from backend.inference.framework_bindings import (ServingDeployment,
                                                  get_binding)
from backend.inference.gpu_specs import get_gpu
from backend.inference.parallel_config import ParallelConfig
from backend.inference.pipeline import analyze
from tests.snapshots import resolve_snapshot, snapshot_config

BATCH = 8


def _config(model: str) -> Dict[str, Any]:
    """The checkpoint's `config.json`, as the snapshot recorded it."""
    return snapshot_config(model)


def _rows(model: str, framework: str, *, parallel: Optional[dict] = None,
          stage: Optional[int] = None) -> List[Dict[str, Any]]:
    """The decode step's per-op rows on an H100, `framework`'s launches."""
    gpu = get_gpu("H100-SXM")
    kwargs = {} if stage is None else {"stage": stage}
    result = analyze(
        resolve_snapshot(model),
        parallel=(ParallelConfig.from_dict(parallel) if parallel
                  else ParallelConfig()),
        framework_binding=get_binding(framework).deployed(
            ServingDeployment.at(gpu_sku="H100-SXM", activation_dtype="bf16")),
        batch_size=BATCH, gpu=gpu, phase="decode",
        avg_input_tokens=1024, avg_output_tokens=64, breakdown=False,
        peak_flops_per_sec=gpu.bf16_tflops * 1e12,
        peak_hbm_bw_bytes_per_sec=gpu.mem_bw_gbps * 1e9, **kwargs)
    # Each row says whether it is a launch across a seam: one whose node
    # names members of the previous layer (`FusedNode.previous_iteration`).
    rows = []
    for row in result.per_op_timing:
        node = result.program.by_uid(int(row["uid"]))
        rows.append({**row, "across_seam": isinstance(node, FusedNode)
                     and bool(node.previous_iteration)})
    return rows


def _seams(rows) -> Dict[str, Dict[str, Any]]:
    """The launches across a seam, by row name."""
    return {row["aggregation_id"]: row for row in rows
            if row["across_seam"]}


def _bpe(config: Dict[str, Any]) -> int:
    return {"bfloat16": 2, "float16": 2, "float32": 4}[config["torch_dtype"]]


# DeepSeek-V3 first: the case whose seam launch also takes the FP8 cast of
# the norm's output, so it is red whether the seam stops fusing or the cast
# stops being absorbed.
@pytest.mark.parametrize("model,framework", [
    ("deepseek-ai/DeepSeek-V3", "vllm"),
    ("meta-llama/Llama-3.2-1B", "sglang"),
    ("meta-llama/Llama-3.2-1B", "vllm"),
])
def test_every_seam_is_one_launch(model: str, framework: str) -> None:
    config = _config(model)
    layers = config["num_hidden_layers"]
    dense = config.get("first_k_dense_replace") or 0
    hidden, bpe = config["hidden_size"], _bpe(config)
    rows = _rows(model, framework)
    seams = _seams(rows)
    exits = [row for name, row in seams.items() if "|" not in name]
    within = [row for name, row in seams.items() if "|" in name]
    # One seam between each two layers, the junction between two
    # segments its own launch, and the last layer's add in the final norm.
    assert [row["count_per_step"] for row in exits] == [1]
    expected = sorted(n for n in ((dense - 1, 1, layers - dense - 1)
                                  if dense else (layers - 1,)) if n)
    assert sorted(row["count_per_step"] for row in within) == expected
    assert all(row["op"] == "fused_add_rmsnorm" for row in seams.values())
    # Every layer's launch within it still fuses: the reads across the seam
    # a matcher now sees refuse no match that has no seam in it.
    within_layer = [row for row in rows
                    if row["aggregation_id"].endswith(
                        "|post_attention_layernorm")]
    assert sum(row["count_per_step"] for row in within_layer) == layers
    # Only layer 0's input norm runs bare: it reads the embedding.
    opening = [row for row in rows
               if row["aggregation_id"].endswith("|input_layernorm")]
    assert sum(row["count_per_step"] for row in opening) == 1
    # Bytes per launch, from the config: the two operands of the add read,
    # the residual and the normed row written, the norm's weight read.
    tokens = BATCH
    for row in within:
        per_launch = row["access_bytes_per_rank"] / row["count_per_step"]
        if row["output_dtype"] == "F8_E4M3":
            # The quantised row, one fp32 scale per 128-wide group.
            block = config["quantization_config"]["weight_block_size"][1]
            written = tokens * hidden + tokens * (hidden // block) * 4
        else:
            written = tokens * hidden * bpe
        assert per_launch == (3 * tokens * hidden * bpe + written
                              + hidden * bpe), row["aggregation_id"]
    if model.startswith("deepseek-ai/"):
        # The cast the GEMM after the norm needs is the launch's own write.
        assert {row["output_dtype"] for row in within} == {"F8_E4M3"}


def test_a_pipeline_cut_is_no_seam() -> None:
    """Under PP=2 the add before the cut runs on the stage that made it and
    the norm after it opens the next stage bare; every other seam fuses."""
    model = "meta-llama/Llama-3.2-1B"
    held = _config(model)["num_hidden_layers"] // 2
    first, last = (_rows(model, "vllm", parallel={"pp": 2}, stage=s)
                   for s in (0, 1))
    for rows in (first, last):
        seams = _seams(rows)
        assert seams["block|input_layernorm__seam"]["count_per_step"] == \
            held - 1
        (opening,) = [row for row in rows
                      if row["aggregation_id"] == "block|input_layernorm"]
        assert opening["count_per_step"] == 1
        # The add the cut keeps standalone is read across a seam elsewhere;
        # no launch within a layer is refused for it.
        (within,) = [row for row in rows if row["aggregation_id"]
                     == "block|post_attention_layernorm"]
        assert within["count_per_step"] == held
    (closing,) = [row for row in first if row["op"] == "add"
                  and row["aggregation_id"].startswith("block|out")]
    assert closing["count_per_step"] == 1
    # A stage's rows hold every node, at the count it runs there (zero for
    # one that runs on another stage).
    assert sum(row["count_per_step"] for row in last if row["op"] == "add"
               and row["aggregation_id"].startswith("block|out")) == 0
    assert _seams(last)["norm__seam@block"]["count_per_step"] == 1


@pytest.mark.regression
def test_an_interleaved_stack_counts_each_junction() -> None:
    """Qwen3-Next interleaves three linear-attention layers with one full
    one: each kind of junction runs as often as the two kinds meet."""
    model = "Qwen/Qwen3-Coder-Next"
    config = _config(model)
    every = config["full_attention_interval"]
    full = [(i + 1) % every == 0 for i in range(config["num_hidden_layers"])]
    pairs = list(pairwise(full))
    meets = {(a, b): sum(1 for p in pairs if p == (a, b))
             for a in (False, True) for b in (False, True)}
    seams = _seams(_rows(model, "vllm"))
    counts = sorted(row["count_per_step"] for name, row in seams.items()
                    if "|" in name)
    assert counts == sorted(n for n in meets.values() if n)


@functools.lru_cache(maxsize=None)
def _two_template_resolved() -> Any:
    """The five-layer DeepSeek-V3 of `_two_template_step`, resolved."""
    from tests.snapshots import shared_loader, snapshot_config

    config = dict(snapshot_config("deepseek-ai/DeepSeek-V3"))
    config.update(
        num_hidden_layers=5, first_k_dense_replace=2, n_routed_experts=8,
        num_experts_per_tok=2, n_group=2, topk_group=1,
        num_nextn_predict_layers=0, hidden_size=256, intermediate_size=512,
        moe_intermediate_size=128, num_attention_heads=4,
        num_key_value_heads=4, q_lora_rank=64, kv_lora_rank=32,
        qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16,
        vocab_size=1024)
    return shared_loader().match(config, model_id="deepseek-ai/DeepSeek-V3")


@functools.lru_cache(maxsize=None)
def _two_template_step() -> Tuple[Any, Any]:
    """A DeepSeek-V3 five layers deep — two dense, three MoE, eight experts
    — at a small width: its dense template's input norm keeps its layer-0
    run beside the seam into layer 1, and its MoE template opens with the
    dense → MoE seam, runs one MoE → MoE seam, and closes into the final
    norm, three classes of one template's iterations
    (`Rewrite.iteration_classes`). The walk and its vLLM serving graph."""
    from backend.inference.serving import rewrite_serving_graph
    from backend.inference.step_eval import evaluate_step

    resolved = _two_template_resolved()
    analysis = evaluate_step(resolved, ParallelConfig(), phase="decode",
                             batch=2, input_tokens=64,
                             output_tokens=8).analysis
    served = rewrite_serving_graph(
        analysis, framework_binding=get_binding("vllm").deployed(
            ServingDeployment.at(gpu_sku="H100-SXM",
                                 activation_dtype="bf16")))
    return analysis, served


def _runs_where_it_runs(program: Any, reader: Any, iterations: Any,
                        writer: Any, how: str) -> None:
    """A launch read at an iteration runs there: the reading layer is where
    a launch across a seam stands (its home), and a launch within one layer
    runs that layer. The iteration is the one of the node whose input it is
    (`Run.at`): a launch reading through a member of the layer before reads
    at that member's. A reader inside a layer reads a launch of some
    layer."""
    if not isinstance(writer, FusedNode):
        return
    runs = set(program.occurrence_layers(writer.uid))
    assert set(iterations) <= runs, (
        how, reader.display_path, sorted(iterations, key=str),
        writer.display_path, sorted(runs, key=str))
    if program.block_of(reader.uid) is not None:
        assert program.block_of(writer.uid) is not None, (
            how, reader.display_path, writer.display_path)


# The edges first: every reader's answer is read off them, so they are the
# one whose defect every other consumer would inherit.
@pytest.mark.parametrize("consumer", ["edges", "traffic", "leaf_input_ids"])
def test_a_reader_reads_the_launch_that_runs_its_iteration(
        consumer: str) -> None:
    """Every reader, at every iteration it runs, resolves each input to a
    launch that runs that iteration (`Edges.runs`, traffic's
    `_ReaderLaunches`, `activation_edges.leaf_input_ids`). A writer taken
    by two launches — the last MoE layer's closing add, by the MoE → MoE
    seam at layer 2 and the exit into the final norm at layer 3 — is read
    from the one that runs the reader's iteration, never the last one
    written nor the one that runs most of them."""
    from backend.arch.eval.activation_edges import (iteration_classes,
                                                    leaf_input_ids)
    from backend.arch.eval.cache import Edges
    from backend.arch.eval.traffic import _ReaderLaunches, _RunLaunches

    analysis, served = _two_template_step()
    program, rewrite = served.program, served.rewrite
    edges = Edges(analysis, program, rewrite)
    leaves = [node for node in program.nodes() if not node.children]
    if consumer == "edges":
        for reader in leaves:
            for index in range(len(reader.inputs_obj or ())):
                for run in edges.runs(reader, index):
                    _runs_where_it_runs(program, reader, (run.at,),
                                        run.node, consumer)
    elif consumer == "traffic":
        # Traffic's launch for each member a reader reads is the one the
        # edges name for that run — including for a model-level reader,
        # which runs at no layer: the launch into the final norm reads the
        # last layer's operands (iteration 3) through its member, and the
        # launch that writes them runs there.
        launches = _RunLaunches(edges, rewrite, program)
        model_level = 0
        for reader in leaves:
            for iterations in launches.classes(reader.uid):
                launch_of = _ReaderLaunches(launches, reader.uid, iterations)
                for index in range(len(reader.inputs_obj or ())):
                    for run in edges.runs(reader, index):
                        if run.iteration not in iterations:
                            continue
                        got = launch_of.get(run.written.uid)
                        want = (run.node if isinstance(run.node, FusedNode)
                                else None)
                        assert got is want, (
                            reader.display_path, run.written,
                            getattr(got, "display_path", None),
                            getattr(want, "display_path", None))
                        if got is not None:
                            _runs_where_it_runs(program, reader, (run.at,),
                                                got, consumer)
                            model_level += run.iteration is None and (
                                run.at is not None)
        assert model_level, "no model-level reader of a layer's run"
    else:
        ids = leaf_input_ids(program, rewrite, edges=edges)
        for reader in leaves:
            for iterations in iteration_classes(program, rewrite, reader.uid):
                for tensor in ids[reader.uid][iterations]:
                    uid = (tensor if isinstance(tensor, int)
                           else int(str(tensor).split("#", 1)[0])
                           if str(tensor).split("#", 1)[0].isdigit()
                           else None)
                    if uid is None:
                        continue
                    _runs_where_it_runs(program, reader, iterations,
                                        program.by_uid(uid), consumer)
    # The case the test exists for is in the graph: one closing add taken
    # by two launches of different iterations.
    assert any(len({rewrite.launch_at(old, it)
                    for it in program.occurrence_layers(old)} - {None}) > 1
               for olds in rewrite.taken.values() for old in olds)


def test_a_seam_moves_no_weight_read() -> None:
    """Fusion moves activation bytes, never weight reads: summed over every
    run, the five-layer DeepSeek-V3 reads the same weights fused across its
    seams as unfused. The dense input norm the launch into layer 1 took one
    run of reads its weight once per run on each of its two rows."""
    from backend.arch.eval.program import Program
    from backend.arch.eval.traffic import Traffic, derive
    from backend.arch.eval.transforms import Deployment
    from backend.arch.eval.fusion import Framework
    from backend.inference.fusion_plans import FUSION_SCHEMES
    from tests.snapshots import shared_walk

    def weight_reads(analysis: Any) -> Any:
        layer = derive(analysis)
        return Traffic.sum(
            value.scaled(analysis.instances.values[uid])
            for uid, value in layer.values.items()
            if not analysis.program.by_uid(uid).children).weight_reads

    deployment = Deployment(Program.of(_two_template_resolved())).at(
        batch=1, input_tokens=256)
    framework = Framework.of(
        get_binding("vllm").deployed(ServingDeployment.at(
            gpu_sku="H100-SXM", activation_dtype="bf16")), FUSION_SCHEMES)
    naive = shared_walk(deployment)
    fused = shared_walk(deployment.fused(framework))
    assert any(isinstance(node, FusedNode) and node.previous_iteration
               for node in fused.program.nodes())
    # The case the test exists for: a node with runs of its own beside a
    # launch that took the others.
    assert any(fused.rewrite.partly(node.uid)
               for node in naive.program.nodes())
    assert weight_reads(naive) == weight_reads(fused)


def _no_seam_launch_takes_a_run_its_own_layer_reads(
        model: str, parallel: dict) -> None:
    """A launch across a seam runs in the reading layer, after the layer its
    previous-layer members belong to, so no other node of that layer may
    read the run it took: that reader would read a tensor not yet written,
    and the activation liveness of its layer would name a tensor no node of
    the layer writes."""
    from backend.arch.eval.cache import Edges
    from backend.inference.activation_peak import activation_peak_at
    from backend.inference.serving import rewrite_serving_graph
    from backend.inference.step_eval import evaluate_step

    step = evaluate_step(
        resolve_snapshot(model), ParallelConfig.from_dict(parallel)
        if parallel else ParallelConfig(), phase="prefill", batch=1,
        input_tokens=512, output_tokens=8)
    binding = get_binding("vllm").deployed(ServingDeployment.at(
        gpu_sku="H100-SXM", activation_dtype="bf16"))
    served = rewrite_serving_graph(step.analysis, framework_binding=binding)
    program, rewrite = served.program, served.rewrite
    taken = {}
    for launch in program.nodes():
        if isinstance(launch, FusedNode) and launch.previous_iteration:
            for member in launch.previous_iteration:
                for at in rewrite.taken[launch.uid][member].values():
                    taken[(member, at)] = launch
    assert taken, "no launch across a seam to check"
    edges = Edges(step.analysis, program, rewrite)
    for reader in program.walked_nodes():
        if reader.children:
            continue
        for index in range(len(reader.inputs_obj or ())):
            for run in edges.runs(reader, index):
                launch = taken.get((run.written.uid, run.written.iteration))
                if launch is None or run.at != run.written.iteration:
                    continue    # not a taken run, or read across the seam
                assert reader.uid == launch.uid, (
                    reader.display_path, run.written, launch.display_path)
    # And the liveness of every layer closes.
    activation_peak_at(step, framework_binding=binding)


def test_a_walked_reader_in_its_own_layer_keeps_its_run() -> None:
    """Step-3.7's MTP layer: its shared-head norm reads the layer's closing
    add inside the layer, a reference the walk bound (`Seams.of`, walked
    readers)."""
    _no_seam_launch_takes_a_run_its_own_layer_reads(
        "stepfun-ai/Step-3.7-Flash", {})


def test_a_grafted_reader_in_its_own_layer_keeps_its_run() -> None:
    """Kimi-K3 at TP 8: in its first (dense) layer an all-reduce a pass
    grafted on the MLP residual reads it inside the layer, a reader the walk
    never bound (`Seams.of`, grafted readers at the runs they declare)."""
    _no_seam_launch_takes_a_run_its_own_layer_reads(
        "moonshotai/Kimi-K3", {"tp_attention": 8, "tp_dense_ffn": 8,
                               "tp_moe_expert": 8})


def test_a_reader_in_some_layers_keeps_only_their_seams(monkeypatch) -> None:
    """A reader inside some layers only — EAGLE-3's auxiliary taps read the
    hidden state of layers 2, L/2 and L-3 — keeps the closing add of those
    layers out of the seam launch, and only of those: every other seam
    fuses, and the layers that differ fall into iteration classes of their
    own. The tap is added to the walk's in-layer reads (`Seams.in_layer`)
    on Llama-3.2-3B; the layers expected are counted from the checkpoint's
    `config.json`, not from the walk."""
    import dataclasses

    from backend.arch.eval import fusion
    from backend.inference.serving import rewrite_serving_graph
    from backend.inference.step_eval import evaluate_step

    model = "meta-llama/Llama-3.2-3B"
    n_layers = _config(model)["num_hidden_layers"]
    taps = {2, n_layers // 2, n_layers - 3}
    binding = get_binding("vllm").deployed(ServingDeployment.at(
        gpu_sku="H100-SXM", activation_dtype="bf16"))

    def served():
        step = evaluate_step(resolve_snapshot(model), ParallelConfig(),
                             phase="decode", batch=2, input_tokens=64,
                             output_tokens=8)
        out = rewrite_serving_graph(step.analysis, framework_binding=binding)
        (launch,) = [node for node in out.program.nodes()
                     if isinstance(node, FusedNode)
                     and node.previous_iteration
                     and out.program.block_of(node.uid) is not None]
        return out, launch

    plain, launch = served()
    (add,) = launch.previous_iteration
    assert set(plain.program.occurrence_layers(launch.uid)) == set(
        range(1, n_layers))
    # The tap: a node of the layer that is no member of the launch.
    members = set(plain.rewrite.origins[launch.uid])
    block = plain.program.block_of(add)
    tap = min(node.uid for node in plain.program.blocks[block].nodes()
              if not node.children and node.uid not in members)

    real = fusion.Seams.of.__func__

    def tapped(cls, *args, **kwargs):
        seams = real(cls, *args, **kwargs)
        return dataclasses.replace(seams, in_layer={
            **seams.in_layer,
            **{(add, it): seams.read_in_layer(add, it) | {tap}
               for it in taps}})

    monkeypatch.setattr(fusion.Seams, "of", classmethod(tapped))
    served_tapped, launch = served()
    program, rewrite = served_tapped.program, served_tapped.rewrite
    # Every seam but the ones after a tapped layer is one launch.
    assert set(program.occurrence_layers(launch.uid)) == (
        set(range(1, n_layers)) - {it + 1 for it in taps})
    # The tapped layers' adds run on their own there, and nowhere else.
    assert set(rewrite.taken[launch.uid][add].values()) == (
        set(range(n_layers - 1)) - taps)
    assert rewrite.partly(add)
