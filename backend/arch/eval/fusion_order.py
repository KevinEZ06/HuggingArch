"""The one rule plan order may not decide: which of two equally specific
plans takes a node set both match.

A binding tries its plans most specific first (`framework_bindings.
match_order`: required pattern nodes, then all pattern nodes, then the
plan's name). The name only makes the order the same in every process; it is
no statement by any author. So where two plans of equal specificity would
take the very same nodes on a graph, the first by name would win silently —
an alphabetical accident priced as a launch. That is refused here, at the
owner of the order (`Framework.rewrite`), with both plans named: the fix is a
pattern that says which one is larger, or a binding that binds one of them.

Which pairs can collide at all is the patterns' own answer
(`NaivePattern.may_take_same_nodes`: equal specificity and an op-compatible
pairing of their nodes); only those pairs are matched on the graph, so a
rewrite pays nothing for the many equally specific plans that share no op.
"""
from __future__ import annotations

from typing import Any, Sequence


def refuse_tied_preemption(name: str, fusions: Sequence[Any], index: int,
                           program: Any, analysis: Any, local_shapes: Any,
                           graft: Any) -> None:
    """Raise when the fusion at `index`, about to apply to `program`, and a
    later one of equal specificity would both take one node set there."""
    first = fusions[index]
    pattern = first.scheme.naive_pattern
    rivals = [later for later in fusions[index + 1:]
              if later.scheme_name != first.scheme_name
              and pattern.may_take_same_nodes(later.scheme.naive_pattern)]
    if not rivals:
        return
    mine = _taken(first, program, analysis, local_shapes, graft)
    for rival in rivals:
        both = mine & _taken(rival, program, analysis, local_shapes, graft)
        if both:
            raise ValueError(
                f"{name}: fusion plans {first.scheme_name!r} and "
                f"{rival.scheme_name!r} are equally specific "
                f"{pattern.specificity()} and both match the nodes "
                f"{sorted(min(both))}; which one takes them would be their "
                f"names' order. Make one pattern larger, or bind one.")


def _taken(fusion: Any, program: Any, analysis: Any, local_shapes: Any,
           graft: Any) -> frozenset:
    """The node sets `fusion` would take on `program`, each as its members."""
    _fused, step = fusion.rewrite(program, analysis, local_shapes, graft)
    return frozenset(frozenset(members) for members in step.origins.values())
