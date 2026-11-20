"""A fused op's input ports are declared once, in its plan's `io_mapping`
(`NaivePattern._check_input_ports`)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from backend.arch.schema.fusion import NaivePattern

NODES = {"add": {"op": "add", "inputs": ["external.x", "external.residual"]},
         "norm": {"op": "rmsnorm", "inputs": ["add.out"]}}


def test_every_external_is_an_input_port_named_for_itself() -> None:
    pattern = NaivePattern(nodes=NODES, io_mapping={
        "residual": "external.residual", "x": "external.x",
        "output": "norm.out"})
    # io_mapping order is the operand order.
    assert pattern.input_ports() == ("residual", "x")


@pytest.mark.parametrize("io_mapping,why", [
    ({"x": "external.x", "output": "norm.out"}, "declares no input port"),
    ({"x": "external.x", "res": "external.residual", "output": "norm.out"},
     "keyed by its external's name"),
    ({"x": "external.x", "residual": "external.residual",
      "bias": "external.bias", "output": "norm.out"}, "no pattern node reads"),
])
def test_a_port_the_plan_does_not_declare_once_is_refused(
        io_mapping, why) -> None:
    with pytest.raises(ValueError, match=why):
        NaivePattern(nodes=NODES, io_mapping=io_mapping)


@pytest.mark.parametrize("shared,why", [
    ({"v": "q"}, "two input ports"),
    ({"x": "x"}, "two input ports"),
    ({"x": "residual", "residual": "x"}, "itself shared"),
])
def test_a_shared_port_names_the_one_it_is_read_as(shared, why) -> None:
    with pytest.raises(ValueError, match=why):
        NaivePattern(nodes=NODES, shared_ports=shared, io_mapping={
            "x": "external.x", "residual": "external.residual",
            "output": "norm.out"})
