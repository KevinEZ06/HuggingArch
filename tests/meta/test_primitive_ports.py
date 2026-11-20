"""A primitive op measured under a signature that reads by port has no ports
of its own; the signature names its operands (`OpSignature.primitive_ports`).
Whether a signature measures a primitive, and how many operands that
primitive takes (`OpSpec.arity`), are the op registry's — which
`backend.inference` may not ask (CLAUDE.md §2.7) — so the join of the two
declarations is held here, where both may be read."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.arch.ops import OP_REGISTRY
from backend.inference.fusion_plans import FUSION_SCHEMES
from backend.inference.op_signatures import (
    get_signature, op_bindings, op_signature)


def _primitives_by_signature():
    """Each signature's primitive ops: registered ops (not a plan's launch)
    measured under it, bound or by their own name."""
    out = {}
    for op in OP_REGISTRY:
        if op in FUSION_SCHEMES and op not in op_bindings():
            continue
        sig = get_signature(op_signature(op))
        if sig is not None:
            out.setdefault(sig.name, []).append(op)
    return out


def test_a_port_reading_signature_names_each_primitive_operand() -> None:
    wrong = []
    for name, ops in sorted(_primitives_by_signature().items()):
        sig = get_signature(name)
        if not sig.keyed_by_port():
            continue
        wrong.extend((name, op, sig.primitive_ports, OP_REGISTRY[op].arity())
                     for op in ops
                     if len(sig.primitive_ports) != OP_REGISTRY[op].arity())
    assert not wrong, (
        "a signature that reads by port measures a primitive whose "
        "operands `primitive_ports` does not name one each: "
        f"{wrong}")


def test_the_gate_sees_the_shared_signatures() -> None:
    # The two signatures a primitive shares with a plan's launch, so the
    # gate above is not vacuous.
    shared = {name for name, ops in _primitives_by_signature().items()
              if get_signature(name).keyed_by_port()}
    assert {"rope", "indexer_logits"} <= shared, shared
