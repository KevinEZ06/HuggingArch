"""Leaf input tensor identities — the one place the activation-liveness edge
join is made.

`backend.inference.activation_peak` reads these instead of re-walking each
leaf's `inputs_obj` itself: eval is the producer of the edge → producer
resolution (traffic prices the same edges for bytes; this names them). The
resolution redirects a producer absorbed into a fused node to that node's
published port, and keeps an alias id for a ref that names an enclosing scope's
input.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Tuple

from .layer import Rewrite


def tensor_id(uid, port: str):
    """Stable identity for one node output tensor. Main output is the uid; a
    named port is `uid#port=` so liveness never rebuilds a path."""
    return uid if not port or port == "out" else f"{uid}#port={port}"


def iteration_classes(program, rewrite: Rewrite, uid: int
                      ) -> Tuple[FrozenSet[Optional[int]], ...]:
    """The iteration classes leaf `uid` runs in (`Rewrite.iteration_classes`
    of its block, at the iterations it runs at); one for a model-level
    leaf."""
    its = frozenset(program.occurrence_layers(uid))
    block = program.block_of(uid)
    if block is None:
        return (its,)
    return tuple(c & its for c in rewrite.iteration_classes(program, block)
                 if c & its)


def leaf_input_ids(program, rewrite: Optional[Rewrite] = None, *,
                   edges=None
                   ) -> Dict[int, Dict[FrozenSet[Optional[int]], List]]:
    """Per leaf uid the step runs, per iteration class it runs in → its
    input tensor identities.

    Each input names what writes the run it reads, the one answer every
    reader asks (`cache.Edges.runs`): the launch that took the producer's
    run, on the port it publishes it at, or the producer itself. A
    template's layers can differ in that — DeepSeek-V3's first MoE layer
    reads the dense → MoE seam's launch, the others the MoE → MoE one — so
    the answer is per class (`iteration_classes`), one launch per input in
    each by construction. A ref whose `src` is None names an enclosing
    scope's input and keeps its `external_sid`. Without `edges` — only for
    a graph no rewrite touched — a producer is its own writer.
    """
    rewrite = rewrite or Rewrite()
    if edges is None and rewrite.origins:
        raise ValueError(
            "a rewritten graph's inputs are answered by the walk's edges "
            "(`cache.Edges.runs`); none were given")
    out: Dict[int, Dict[FrozenSet[Optional[int]], List]] = {}
    # The leaves the step runs: a stack the walk does not (an encoder, in
    # the language model's step) runs at no iteration here.
    for node in program.walked_nodes():
        if node.children:
            continue
        refs = tuple(node.inputs_obj or ())
        runs = ([edges.runs(node, index) if ref.src is not None else ()
                 for index, ref in enumerate(refs)]
                if edges is not None else None)
        per_class: Dict[FrozenSet[Optional[int]], List] = {}
        for iterations in iteration_classes(program, rewrite, node.uid):
            inputs: List = []
            for index, ref in enumerate(refs):
                if ref.src is None:
                    inputs.append(ref.external_sid)
                    continue
                if runs is None:
                    producer, port = program.output_leaf(
                        ref.src, ref.port or "out")
                    inputs.append(tensor_id(producer.uid, port))
                    continue
                if not runs[index] and (
                        ref.src.uid in edges.handed_exits()):
                    # An encoder's output handed over: no node of this
                    # step writes it, and it is its producer's tensor.
                    producer, port = program.output_leaf(
                        ref.src, ref.port or "out")
                    inputs.append(tensor_id(producer.uid, port))
                    continue
                written = {tensor_id(run.node.uid, run.port or "out")
                           for run in runs[index]
                           if run.iteration in iterations}
                if len(written) != 1:
                    raise ValueError(
                        f"{node.display_path}: input {index} reads "
                        f"{sorted(map(str, written))!r} in one iteration "
                        f"class; one class is one set of launches")
                inputs.append(written.pop())
            per_class[iterations] = inputs
        out[node.uid] = per_class
    return out
