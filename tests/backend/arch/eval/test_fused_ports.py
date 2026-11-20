"""Which tensor is which operand of a fused launch is its plan's declaration
(`NaivePattern.input_ports`, the `external.<name>` of `io_mapping`), named on
the launch (`FusedNode.operand_ports`) — not the order its members' uids sort
in, which differs between a layer and a seam, and between two layers'
templates.

The oracle walks the other way round: from the plan's pattern and the
binding the launch records (`FusedNode.ports`, pattern node → member), each
port's tensor is the input the member that declares `external.<port>`
reads. The launch's operand that names the port must be that tensor, by
identity (its producer and port, or the scope alias it reads), not by rank:
two operands of one rank swapped would pass a rank check.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from backend.arch.eval.fusion import FusedProgram
from backend.arch.eval.launch import FusedNode
from backend.inference.framework_bindings import (
    ServingDeployment, get_binding)
from backend.inference.fusion_plans import FUSION_SCHEMES
from backend.inference.parallel_config import ParallelConfig
from tests.snapshots import resolve_snapshot


@functools.lru_cache(maxsize=None)
def _served(model: str, phase: str = "decode") -> Any:
    from backend.inference.serving import rewrite_serving_graph
    from backend.inference.step_eval import evaluate_step

    step = evaluate_step(resolve_snapshot(model), ParallelConfig(),
                         phase=phase, batch=2, input_tokens=256,
                         output_tokens=8)
    return rewrite_serving_graph(
        step.analysis, framework_binding=get_binding("vllm").deployed(
            ServingDeployment.at(gpu_sku="H100-SXM", activation_dtype="bf16")))


def _launches(served: Any):
    """`(launch, the pass's rewrite, the graph it rewrote)` for every fused
    launch of the serving graph, each read off the pass that made it."""
    program = served.program
    while isinstance(program, FusedProgram):
        for launch in program.fused:
            if launch.operand_ports:
                yield launch, program.rewrite, program.base
        program = program.base


def _tensor(graph: Any, uid: int, index: int) -> Tuple[Any, ...]:
    """The tensor input `index` of node `uid` reads: its producer and port,
    or the alias of the scope input it names."""
    ref = tuple(graph.by_uid(uid).inputs_obj)[index]
    if ref.src is not None:
        return (ref.src.uid, ref.port or "out")
    return ("alias", ref.external_sid)


def _declared(launch: FusedNode, graph: Any) -> Dict[str, set]:
    """Per port, the tensors the plan says fill it: the inputs, at
    `external.<port>`, of the members the launch bound."""
    pattern = FUSION_SCHEMES[launch.op].naive_pattern
    out: Dict[str, set] = {}
    if pattern.parallel is not None:
        names = [r.partition(".")[2] for r in pattern.parallel.shared_inputs]
        for uid, _port in launch.ports.values():
            for index, name in enumerate(names):
                out.setdefault(name, set()).add(_tensor(graph, uid, index))
        return out
    for node_name, (uid, _port) in launch.ports.items():
        spec = pattern.nodes[node_name]
        arity = len(tuple(graph.by_uid(uid).inputs_obj or ()))
        for index, ref in enumerate((*spec.inputs, *spec.optional_inputs)):
            head, _, name = ref.partition(".")
            if head == "external" and index < arity:
                out.setdefault(name, set()).add(_tensor(graph, uid, index))
    return out


def _named(launch: FusedNode, rewrite: Any, graph: Any) -> Dict[str, Any]:
    """Per port, the operand the launch names it on."""
    out: Dict[str, Any] = {}
    for position, ports in enumerate(launch.operand_ports):
        member, index = rewrite.inputs[(launch.uid, position)]
        for port in ports:
            assert port not in out, (launch.display_path, port)
            out[port] = _tensor(graph, member, index)
    return out


# Qwen3.8-Flash-Next first: its `hc_combine_norm` runs in a layer and across
# every seam, whose reading member's uid sorts before the previous layer's
# write — the case operand order by uid got wrong. DeepSeek-V4.1-Flash: a K
# that is its V, an absent mask, and query/latent pairs in either graph
# order. gemma-4: a K that is its V.
@pytest.mark.parametrize("model", [
    "Qwen/Qwen3.8-Flash-Next",
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "google/gemma-4-12B-it",
])
def test_each_port_is_the_tensor_its_member_reads(model: str) -> None:
    served = _served(model)
    checked = set()
    for launch, rewrite, graph in _launches(served):
        declared = _declared(launch, graph)
        named = _named(launch, rewrite, graph)
        # Order: the plan's port order, whatever the members' uids.
        order = FUSION_SCHEMES[launch.op].naive_pattern.input_ports()
        firsts = [min(order.index(p) for p in ports)
                  for ports in launch.operand_ports]
        assert firsts == sorted(firsts), (launch.display_path,
                                          launch.operand_ports)
        for port, tensor in named.items():
            # A crossed formal is no operand; every port named is one
            # tensor its declaring members read.
            assert declared.get(port) == {tensor}, (
                launch.display_path, port, tensor, declared.get(port))
        checked.add((launch.op, bool(launch.previous_iteration)))
    assert checked
    if model == "Qwen/Qwen3.8-Flash-Next":
        # The same op in a layer and across a seam, both checked.
        assert {("hc_combine_norm", False),
                ("hc_combine_norm", True)} <= checked


def test_one_tensor_that_is_two_ports_is_one_operand_of_both() -> None:
    """DeepSeek-V4's attention reads one latent as its K and its V: one
    operand (traffic reads it once), named for both ports, which a
    signature reads at either."""
    served = _served("deepseek-ai/DeepSeek-V4.1-Flash")
    merged = [launch for launch, _r, _g in _launches(served)
              if launch.op == "flash_attention"
              and ("k", "v") in launch.operand_ports]
    assert merged
    for launch in merged:
        assert len(launch.inputs_obj) == len(launch.operand_ports)


def test_a_port_no_tensor_fills_is_absent_and_the_rest_keep_theirs() -> None:
    """Qwen3.8's indexer weighs no heads (`w` absent): the launch names q
    and k, and the signature reads them where they are."""
    from backend.inference.op_signatures import get_signature, op_signature

    served = _served("Qwen/Qwen3.8-Flash-Next")
    (ports,) = {launch.operand_ports for launch, _r, _g in _launches(served)
                if launch.op == "indexer_logits"}
    assert ports == (("q",), ("k",))
    sig = get_signature(op_signature("indexer_logits"))
    placed = sig.at_ports(ports).engine_inputs
    # q and k where the launch has them; no layout read off a `w` it has
    # not.
    assert {pos for pos, _layout in placed} == {0, 1}
    assert len(placed) == len([key for key, _l in sig.engine_inputs
                               if key != "w"])
