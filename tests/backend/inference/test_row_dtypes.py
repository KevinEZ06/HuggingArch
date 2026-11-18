"""Each per-op row states the element dtype of every tensor its shape
columns name (`input_dtypes` per `shape_in` input, `output_dtype` for
`shape_out`, `port_dtypes` per `shape_outs` port), as the walk billed the
read or write — so the per-op table can show a format change, which the
shapes alone never do."""
from __future__ import annotations

import pytest

from backend.inference.capacity import estimate_capacity
from backend.inference.parallel_config import ParallelConfig
from tests.snapshots import resolve_snapshot, snapshot_config

LLAMA = "meta-llama/Llama-3.2-3B"


@pytest.fixture(scope="module")
def decode_rows():
    est = estimate_capacity(
        config=snapshot_config(LLAMA), model_id=LLAMA,
        resolved=resolve_snapshot(LLAMA), gpu_sku="H100-SXM",
        parallel=ParallelConfig(), num_gpus=1, gpu_memory_fraction=0.9,
        avg_input_tokens=1024, avg_output_tokens=64,
        prefill_batch_size=2, decode_batch_size=2,
        selection_mode="theoretical_only", measurement_scope=None,
        precision={"preset": "w_fp8_kv_fp8"})
    return est["per_op_timing_decode"]


def _row(rows, path):
    return next(r for r in rows if r.get("display_path") == path)


def test_an_fp8_gemm_reads_fp8_and_writes_the_activation_dtype(decode_rows):
    """Under FP8 weights the projection's kernel eats its operand in FP8
    (the quantised copy the row bills) and writes BF16: the two sides of
    the shape column differ in dtype where they agree in nothing else."""
    o_proj = _row(decode_rows, "self_attn.o_proj")
    assert o_proj["input_dtypes"] == ["F8_E4M3"]
    assert o_proj["output_dtype"] == "BF16"


def test_every_input_has_its_own_slot(decode_rows):
    """One dtype per input tensor, in `shape_in` order: a residual add
    states both operands, not the first one's for the pair."""
    add = _row(decode_rows, "out__block")
    assert add["input_dtypes"] == ["BF16", "BF16"]
    for row in decode_rows:
        shape_in = row.get("shape_in")
        if not shape_in:
            assert row.get("input_dtypes") is None, row.get("display_path")
            continue
        arity = len(shape_in) if isinstance(shape_in[0], list) else 1
        assert len(row["input_dtypes"]) == arity, row.get("display_path")


def test_the_weight_operand_is_named_with_its_shape_and_dtype(decode_rows):
    """A projection's own weight is an operand the activation columns never
    name: the row lists it, as one rank holds it, at its resident dtype —
    the FP8 the preset stores it at, not the activation's BF16."""
    o_proj = _row(decode_rows, "self_attn.o_proj")
    assert [(w["shape"], w["dtype"]) for w in o_proj["weights"]] == [
        ([3072, 3072], "F8_E4M3")]
    assert o_proj["weights"][0]["name"].endswith("o_proj.weight")


MLA = "deepseek-ai/DeepSeek-V2-Lite"


def test_a_split_piece_is_named_as_the_piece_it_streams():
    """Absorbed MLA's `q_absorbed_nope` streams W_UK, the piece split out
    of `kv_b_proj` at load and dequantised to BF16 (vLLM `split_kv_b_proj`):
    its row names that piece — the stored tensor's name, the rows of it the
    piece covers, BF16 — and the row's `weight_dtype` and weight bytes are
    that same tensor's. The preset stores `kv_b_proj` FP8, so a row that
    named the stored tensor would disagree with its own bytes. The shape is
    worked out here from the checkpoint config (heads × qk_nope rows of
    kv_lora columns), not from the walk."""
    config = snapshot_config(MLA)
    est = estimate_capacity(
        config=config, model_id=MLA,
        resolved=resolve_snapshot(MLA), gpu_sku="H100-SXM",
        parallel=ParallelConfig(), num_gpus=1, gpu_memory_fraction=0.9,
        avg_input_tokens=1024, avg_output_tokens=64,
        prefill_batch_size=2, decode_batch_size=2,
        selection_mode="theoretical_only", measurement_scope=None,
        precision={"preset": "w_fp8_kv_fp8"})
    rows = [r for r in est["per_op_timing_decode"]
            if r.get("display_path") == "self_attn.q_absorbed_nope"]
    assert rows
    piece = [config["num_attention_heads"] * config["qk_nope_head_dim"],
             config["kv_lora_rank"]]
    for row in rows:
        (weight,) = row["weights"]
        assert weight["name"].endswith("kv_b_proj.weight")
        assert (weight["shape"], weight["dtype"]) == (piece, "BF16")
        assert row["weight_dtype"] == weight["dtype"]
        assert row["weight_read_bytes_per_rank"] == (
            piece[0] * piece[1] * 2 * row["count_per_step"])
        assert not row["metadata_bytes_per_rank"]


def test_a_split_pieces_shards_are_the_piece_it_streams():
    """The scale grid a row publishes (`weight_shards`, what Kernel Bench's
    block-scale refusal reads) describes the same tensor as its weights
    column: W_UK, BF16, with no scale grid — not the FP8 `kv_b_proj` it is
    split out of, whose per-tile blocks and whole rows the shards used to
    report beside a BF16 weight. Under TP 2, so the per-rank rows are the
    piece's share of the rank's heads, not of the stored tensor. The piece
    is worked out here from the checkpoint config, not from the walk."""
    config = snapshot_config(MLA)
    tp = 2
    est = estimate_capacity(
        config=config, model_id=MLA,
        resolved=resolve_snapshot(MLA), gpu_sku="H100-SXM",
        parallel=ParallelConfig(tp_attention=tp), num_gpus=tp,
        gpu_memory_fraction=0.9,
        avg_input_tokens=1024, avg_output_tokens=64,
        prefill_batch_size=2, decode_batch_size=2,
        selection_mode="theoretical_only", measurement_scope=None,
        precision={"preset": "w_fp8_kv_fp8"})
    rows = [r for r in est["per_op_timing_decode"]
            if r.get("display_path") == "self_attn.q_absorbed_nope"]
    assert rows
    heads, nope, lora = (config["num_attention_heads"],
                         config["qk_nope_head_dim"], config["kv_lora_rank"])
    for row in rows:
        (weight,) = row["weights"]
        (shard,) = row["weight_shards"]
        assert weight["shape"] == [heads // tp * nope, lora]
        assert (shard["rows"], shard["width"]) == (heads * nope, lora)
        assert ([shard["rows_local"], shard["width_local"]]
                == weight["shape"])
        assert weight["dtype"] == "BF16"
        assert shard["granularity"] is None
