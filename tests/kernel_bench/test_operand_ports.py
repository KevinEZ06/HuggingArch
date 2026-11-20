"""A job names the plan port each operand of a fused launch fills
(`operand_ports`), and a signature that reads a multi-port launch's operands
by port refuses a job that names none — one planned before launches named
their operands, whose operand order was the members' uid order — rather than
reading its tensors by where they stand. A single-port launch's one operand
is position 0 at any age."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kernel_bench.invocation import Invocation
from kernel_bench.refs import _carried_axes

#: Qwen3.8-Flash-Next's hc_combine_norm at prefill, in its plan's port order.
HC_COMBINE = {
    "op": "hc_combine_norm", "signature": "hc_combine_norm",
    "dims": {"B": 1, "s_query": 2048, "n_hidden_copies": 4, "d_model": 2560},
    "dtype_env": {"activation": "bf16"}, "phase": "prefill",
    "shape_in": [[1, 2048, 4], [1, 2048, 2560], [1, 2048, 4, 2560], [10240]],
}
PORTS = [["injection"], ["block"], ["residual"], ["weight"]]


def test_a_multi_port_job_reads_its_operands_by_the_ports_it_names() -> None:
    inv = Invocation.from_job({**HC_COMBINE, "operand_ports": PORTS})
    assert _carried_axes(inv) == {"B": 1, "s_query": 2048,
                                  "n_hidden_copies": 4, "d_model": 2560}


def test_a_multi_port_job_that_names_no_ports_is_refused() -> None:
    inv = Invocation.from_job(HC_COMBINE)
    with pytest.raises(ValueError, match="names none"):
        _carried_axes(inv)


def test_a_single_port_job_of_any_age_reads_its_one_operand() -> None:
    inv = Invocation.from_job({
        "op": "hc_silu", "signature": "hc_silu",
        "dims": {"B": 1, "s_query": 16, "d_hc_mix": 320},
        "dtype_env": {"activation": "bf16"}, "phase": "prefill",
        "shape_in": [1, 16, 320]})
    assert inv.operand_ports is None
    assert _carried_axes(inv) == {"B": 1, "s_query": 16, "d_hc_mix": 320}


def test_one_tensor_of_two_ports_is_read_as_the_first() -> None:
    """Hy4's sparse sink span reads its stored latent as its K and its V —
    one operand of both ports, its K's 576 columns. The V port reads the
    leading 512 of the same storage, so its layout is not laid on that
    tensor: the cell's `d_v_head` is the output's (`_dsa_identity`), and
    the harness keeps the job's, never the latent's width."""
    inv = Invocation.from_job({
        "op": "flash_dsa_sink", "signature": "mla_sparse_sink_attention",
        "dims": {"B": 2, "n_heads": 64, "s_query": 1, "s_kv": 1088,
                 "n_topk_keys": 1088, "d_head": 576, "d_v_head": 512,
                 "kv_block_size": 64},
        "dtype_env": {"activation": "bf16", "kv": "bf16"},
        "phase": "decode",
        "shape_in": [[2, 64, 1, 576], [2, 1088, 576], [64], [2, 1, 1088]],
        "operand_ports": [["q"], ["k", "v"], ["sinks"], ["mask"]]})
    carried = _carried_axes(inv)
    assert carried["d_head"] == 576 and "d_v_head" not in carried, carried


def test_a_partial_rope_reads_the_head_split_query_and_key() -> None:
    """MiniMax-M3's norm + partial-rope launch: its ports are the head-split
    Q and K the norms read, which key the cell; the packed [q|k|v] row is
    the kernels' argument, built from the cell."""
    import json

    from backend.inference.benchmark_identity import (
        identity_shape_from_shape_in)
    got = identity_shape_from_shape_in(
        "fused_qk_norm_partial_rope", [[2, 48, 128, 64], [2, 4, 128, 64]],
        spec_dims={"d_rope": 32}, operand_ports=[["q"], ["k"]])
    assert json.loads(got) == {"B": 2, "s_query": 128, "n_heads": 48,
                               "n_kv_heads": 4, "d_head": 64, "d_rope": 32}
