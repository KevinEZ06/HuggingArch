"""How a signature names the operands it reads (`op_signatures.
_check_operand_keys`): a fused launch of several ports by port, a single
operand — one port, a collective's flow — by position 0, and a primitive op
measured under a port-keyed signature by the ports `primitive_ports` gives
its operands."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.inference.op_signatures import (
    _check_operand_keys, get_signature)


def _with(name, **changes):
    return dataclasses.replace(get_signature(name), **changes)


def test_the_catalog_names_every_operand_one_way() -> None:
    for name in ("hc_combine_norm", "rope", "indexer_logits",
                 "allreduce_fused_rmsnorm", "gemm"):
        _check_operand_keys(get_signature(name))


def test_a_position_on_a_multi_port_launch_is_refused() -> None:
    sig = get_signature("hc_combine_norm")
    by_position = tuple((i, layout) for i, (_port, layout)
                        in enumerate(sig.engine_inputs))
    with pytest.raises(ValueError, match="name the port"):
        _check_operand_keys(_with("hc_combine_norm",
                                  engine_inputs=by_position))


def test_a_port_on_a_collective_is_refused() -> None:
    sig = get_signature("allreduce_fused_rmsnorm")
    by_port = tuple(("partial", layout) for _pos, layout in sig.engine_inputs)
    with pytest.raises(ValueError, match="flow its collective moves"):
        _check_operand_keys(_with("allreduce_fused_rmsnorm",
                                  engine_inputs=by_port))


def test_a_primitive_operand_names_a_port_the_plan_declares() -> None:
    with pytest.raises(ValueError, match="no input port"):
        _check_operand_keys(_with("rope", primitive_ports=("q",)))
    with pytest.raises(ValueError, match="read by position"):
        _check_operand_keys(_with("gemm", primitive_ports=("x",)))
