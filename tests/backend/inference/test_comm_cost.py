"""Tests for the parallelism comm cost — the collective per-rank
traffic formulas and their end-to-end wiring into the per-op roofline.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.comm_cost import (
    COMM_OPS, COMM_REACH_HOST, COMM_REACH_SCALEOUT, COMM_REACH_SCALEUP,
    Fabric, collective_latency, collective_per_rank_bytes,
    collective_phases, comm_reach, price_collective,
)
from backend.inference.op_signatures import COMM_DOMAIN_AXIS


class TestCollectiveFormulas(unittest.TestCase):
    def test_world_one_is_zero(self):
        # No peers → nothing crosses the interconnect.
        for op in COMM_OPS:
            self.assertEqual(collective_per_rank_bytes(op, 1, 1000), 0.0)

    def test_ring_factors(self):
        # all_reduce = 2(N-1)/N · bytes; the others = (N-1)/N · bytes.
        N, B = 8, 1000.0
        self.assertAlmostEqual(
            collective_per_rank_bytes("all_reduce", N, B), 2 * 7 / 8 * B)
        for op in ("reduce_scatter", "all_gather", "all_to_all"):
            self.assertAlmostEqual(
                collective_per_rank_bytes(op, N, B), 7 / 8 * B)

    def test_all_reduce_is_rs_plus_ag(self):
        # The headline SP equivalence: a ring all_reduce moves exactly as
        # much as a reduce_scatter + all_gather.
        N, B = 8, 1000.0
        ar = collective_per_rank_bytes("all_reduce", N, B)
        rs = collective_per_rank_bytes("reduce_scatter", N, B)
        ag = collective_per_rank_bytes("all_gather", N, B)
        self.assertAlmostEqual(ar, rs + ag)

    def test_non_comm_op_zero(self):
        self.assertEqual(collective_per_rank_bytes("linear", 8, 1000), 0.0)


class TestCollectivePhases(unittest.TestCase):
    """Hierarchy-aware decomposition. Whose contract: at most one phase
    when the collective fits inside a single NVLink domain; for cross-
    domain all_reduce the NCCL/Megatron 3-phase schedule
    (intra-RS + inter-AR-on-shard + intra-AG)."""

    def test_intra_only_single_phase(self):
        # world fits inside one node → single intra-fabric phase.
        ph = collective_phases("all_reduce", 8, 1024, gpus_per_node=8)
        self.assertEqual(len(ph), 1)
        self.assertEqual(ph[0]["name"], "all_reduce")
        self.assertEqual(ph[0]["fabric"], "intra")
        self.assertEqual(ph[0]["world"], 8)
        # Ring AR per-rank = 2(N-1)/N · T.
        self.assertAlmostEqual(ph[0]["bytes_per_rank"], 2 * 7 / 8 * 1024)

    def test_cross_node_all_reduce_is_three_phase(self):
        # world=16 > gpus_per_node=8, world % gpus_per_node == 0:
        # 3-phase hierarchical AR.
        T = 8192.0
        ph = collective_phases("all_reduce", 16, T, gpus_per_node=8)
        self.assertEqual([p["name"] for p in ph],
                         ["reduce_scatter", "all_reduce", "all_gather"])
        self.assertEqual([p["fabric"] for p in ph],
                         ["intra", "inter", "intra"])
        self.assertEqual([p["world"] for p in ph], [8, 2, 8])
        # Phase 1 (intra RS on full T): (8-1)/8 · T.
        self.assertAlmostEqual(ph[0]["bytes_per_rank"], 7 / 8 * T)
        # Phase 2 (inter AR on shard T/N_intra): 2(N_inter-1)/N_inter · T/8.
        self.assertAlmostEqual(ph[1]["bytes_per_rank"], 2 * 1 / 2 * (T / 8))
        # Phase 3 (intra AG on full T): (8-1)/8 · T.
        self.assertAlmostEqual(ph[2]["bytes_per_rank"], 7 / 8 * T)

    def test_cross_node_non_ar_falls_back_to_flat_ring_on_inter(self):
        # RS / AG / A2A don't have a widely-deployed hierarchical
        # decomposition in NCCL — they take a single inter-fabric ring.
        for op in ("reduce_scatter", "all_gather", "all_to_all"):
            ph = collective_phases(op, 16, 8192, gpus_per_node=8)
            self.assertEqual(len(ph), 1)
            self.assertEqual(ph[0]["fabric"], "inter")
            self.assertEqual(ph[0]["world"], 16)
            self.assertAlmostEqual(ph[0]["bytes_per_rank"], 15 / 16 * 8192)

    def test_non_divisible_world_falls_back_to_flat_ring(self):
        # When world is not a multiple of gpus_per_node (e.g. EP=12 on
        # 8-GPU nodes) the hierarchical decomposition isn't well-defined;
        # fall back to a single inter-fabric ring.
        ph = collective_phases("all_reduce", 12, 8192, gpus_per_node=8)
        self.assertEqual(len(ph), 1)
        self.assertEqual(ph[0]["fabric"], "inter")
        self.assertEqual(ph[0]["world"], 12)

    def test_world_one_returns_empty(self):
        # No peers → no phases at all.
        for op in COMM_OPS:
            self.assertEqual(collective_phases(op, 1, 1000, gpus_per_node=8), [])

    # --- base_stride (mesh topology of composed parallelism) -------------
    def test_base_stride_1_matches_legacy_size_only(self):
        # base_stride defaults to 1; passing it explicitly must reproduce
        # the size-only behavior byte-for-byte across the fabric cases.
        for op in ("all_reduce", "reduce_scatter", "all_gather", "all_to_all"):
            for n in (4, 8, 16, 12):
                self.assertEqual(
                    collective_phases(op, n, 8192, gpus_per_node=8),
                    collective_phases(op, n, 8192, gpus_per_node=8, base_stride=1),
                    f"{op} world={n} drifted when base_stride=1 passed explicitly",
                )

    def test_outer_small_axis_takes_the_inter_fabric(self):
        # THE FIX: cp=4 composed over tp_attention=8 on an 8-GPU node.
        # Each cp member is a whole TP inner_group (8 ranks) away → one member
        # per node → fully cross-node, even though 4 <= gpus_per_node=8.
        # Size-only logic mis-prices this as a single intra ring.
        intra = collective_phases("all_to_all", 4, 8192, gpus_per_node=8)
        self.assertEqual(intra[0]["fabric"], "intra")  # legacy (wrong) verdict
        topo = collective_phases(
            "all_to_all", 4, 8192, gpus_per_node=8, base_stride=8)
        self.assertEqual(len(topo), 1)
        self.assertEqual(topo[0]["fabric"], "inter")   # corrected verdict
        self.assertEqual(topo[0]["world"], 4)

    def test_inner_composed_axis_stays_intra(self):
        # ep=2 nested inside tp_moe_expert=4 on an 8-GPU node: the two
        # experts × 4-way TP = 8 ranks = exactly one node, so the EP
        # collective is node-local → intra (matches the DSv3 golden case).
        ph = collective_phases(
            "all_to_all", 2, 4096, gpus_per_node=8, base_stride=4)
        self.assertEqual(len(ph), 1)
        self.assertEqual(ph[0]["fabric"], "intra")

    def test_straddling_ar_uses_per_node_group_in_hierarchy(self):
        # AR over world=4 with base_stride=4 on g=8: members_per_node =
        # 8//4 = 2, so the group straddles (2 nodes × 2 members). The
        # hierarchical AR's intra phases run on the 2-member per-node
        # group, NOT on gpus_per_node.
        ph = collective_phases(
            "all_reduce", 4, 16384, gpus_per_node=8, base_stride=4)
        self.assertEqual([p["fabric"] for p in ph], ["intra", "inter", "intra"])
        self.assertEqual(ph[0]["world"], 2)   # n_intra = members_per_node
        self.assertEqual(ph[1]["world"], 2)   # n_inter = world / n_intra

    def test_a_pipeline_send_is_one_hop_of_the_whole_tensor(self):
        # A send across a pipeline cut moves the sender's tensor once,
        # whatever the pp degree: intra-node when every stage's counterpart
        # (base_stride ranks apart) shares the node, inter otherwise.
        self.assertEqual(collective_per_rank_bytes("send_recv", 4, 1000.0),
                         1000.0)
        (near,) = collective_phases(
            "send_recv", 4, 1000.0, gpus_per_node=8, base_stride=2)
        self.assertEqual((near["fabric"], near["bytes_per_rank"]),
                         ("intra", 1000.0))
        (far,) = collective_phases(
            "send_recv", 4, 1000.0, gpus_per_node=8, base_stride=4)
        self.assertEqual((far["fabric"], far["bytes_per_rank"]),
                         ("inter", 1000.0))


class TestCommCostInPipeline(unittest.TestCase):
    """End-to-end: analyze() attaches a t_comm rooftop term to the comm
    nodes, scaled by the layer count and the right TP world size."""

    @classmethod
    def setUpClass(cls):
        from backend.analyzer import fetch_config
        from tests.snapshots import shared_loader
        from backend.inference.framework_bindings import get_binding
        cls.resolved = shared_loader().match(fetch_config("meta-llama/Llama-2-7b-hf"), model_id="meta-llama/Llama-2-7b-hf")
        cls.binding = get_binding("vllm")

    def _rows(self, **pc_kwargs):
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        res = analyze(
            self.resolved, parallel=ParallelConfig(**pc_kwargs),
            framework_binding=self.binding, batch_size=1, avg_input_tokens=4096, avg_output_tokens=0,
            peak_flops_per_sec=1e15, peak_hbm_bw_bytes_per_sec=3.35e12,
            interconnect_bw_bytes_per_sec=900e9,
            breakdown=True,
        )
        return res.per_op_timing

    def test_tp_comm_has_time_and_scales_by_layers(self):
        # Per-layer collectives only — the vocab-parallel embed all_reduce /
        # lm_head all_gather are top-level (count_per_step=1, once per forward),
        # not multiplied through the 32 layers, so they're scoped out of the
        # "scales by layers" assertion below.
        rows = {r["op"]: r for r in self._rows(tp_attention=8, tp_dense_ffn=8)
                if r.get("comm_bytes_per_rank", 0) > 0
                and "vocab_" not in str(r.get("aggregation_id", ""))}
        # Both the bare FFN-exit all_reduce and the fused attention norm
        # (which absorbed its all_reduce) carry comm.
        self.assertIn("all_reduce", rows)
        self.assertIn("allreduce_fused_rmsnorm", rows)
        for r in rows.values():
            self.assertGreater(r["t_comm_s"], 0)
            # llama has 32 layers — the per-step comm is multiplied through.
            self.assertEqual(r["count_per_step"], 32)

    def test_no_comm_no_cost_at_tp1(self):
        rows = self._rows(tp_attention=1, tp_dense_ffn=1)
        self.assertFalse(any(r.get("comm_bytes_per_rank", 0) > 0 for r in rows))

    def _dsv3_comm(self, **pc_kwargs):
        from backend.analyzer import fetch_config
        from tests.snapshots import shared_loader
        from backend.inference.framework_bindings import get_binding
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        resolved = shared_loader().match(fetch_config("deepseek-ai/DeepSeek-V3"), model_id="deepseek-ai/DeepSeek-V3")
        res = analyze(
            resolved, parallel=ParallelConfig(**pc_kwargs),
            framework_binding=get_binding("vllm"),
            batch_size=1, avg_input_tokens=4096, avg_output_tokens=0,
            peak_flops_per_sec=1e15, peak_hbm_bw_bytes_per_sec=3.35e12,
            interconnect_bw_bytes_per_sec=900e9,
            breakdown=True,
        )
        return res.per_op_timing

    def test_ep_derives_all_to_all_when_the_resting_cut_splits_tokens(self):
        # Expert parallelism's all_to_all pair is a state transition, not an
        # op-intrinsic graft: it exists because the tokens arrive grouped by
        # REQUEST (DP-attention) and the expert bank wants them grouped by
        # OWNER, one tensor cut two ways over one process group. Both legs flow
        # through the generic comm cost path.
        rows = self._dsv3_comm(ep_moe=8, dp_attention=8)
        a2a_rows = [r for r in rows if r["op"] == "all_to_all"
                    and r.get("comm_bytes_per_rank", 0) > 0]
        # One from the permute boundary, one from the combine boundary.
        self.assertGreaterEqual(len(a2a_rows), 2)
        for r in a2a_rows:
            self.assertGreater(r["t_comm_s"], 0)
        # Permute itself should carry zero FLOPs / zero comm bytes —
        # purely a shape op.
        permute_rows = [r for r in rows if r["op"] == "permute"]
        self.assertTrue(permute_rows, "expected permute rows in DSv3 MoE breakdown")
        for r in permute_rows:
            self.assertEqual(r.get("comm_bytes_per_rank", 0), 0)

    def test_ep_without_a_token_cut_reduces_instead_of_exchanging(self):
        # The same degree of expert parallelism on a deployment that does NOT
        # cut tokens is a different bill, and the difference is not a special
        # case: every rank holds every token, so nothing re-partitions on the
        # way in, and each token leaves the bank with a partial sum over the
        # experts its rank owns. One reduction, no exchange — what a TP+EP
        # deployment without DP-attention actually runs.
        rows = self._dsv3_comm(ep_moe=8)
        self.assertFalse(any(r["op"] == "all_to_all"
                             and r.get("comm_bytes_per_rank", 0) > 0
                             for r in rows))
        reduced = [r for r in rows if r["op"] == "all_reduce"
                   and r.get("comm_bytes_per_rank", 0) > 0]
        self.assertTrue(reduced, "the routed partial has to reduce somewhere")
        for r in reduced:
            self.assertGreater(r["t_comm_s"], 0)

    def test_ep_one_derives_nothing(self):
        # ep=1 owns every expert, so neither regime has anything to pay for.
        rows = self._dsv3_comm(ep_moe=1)
        self.assertFalse(any(r.get("comm_bytes_per_rank", 0) > 0 for r in rows))

    def test_no_interconnect_bw_degrades_to_zero_time(self):
        # Without an interconnect bandwidth, comm bytes are still sized
        # but t_comm is 0 (structural-only, the pre-cost behavior).
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        res = analyze(
            self.resolved, parallel=ParallelConfig(tp_attention=8),
            framework_binding=self.binding, batch_size=1, avg_input_tokens=4096, avg_output_tokens=0,
            peak_flops_per_sec=1e15, peak_hbm_bw_bytes_per_sec=3.35e12,
            interconnect_bw_bytes_per_sec=None,
            breakdown=True,
        )
        comm = [r for r in res.per_op_timing if r.get("comm_bytes_per_rank", 0) > 0]
        self.assertTrue(comm)
        self.assertTrue(all(r["t_comm_s"] == 0 for r in comm))


class TestCommBandwidthLaneSplit(unittest.TestCase):
    """End-to-end check on the intra (NVLink) vs inter (RDMA) bandwidth
    selection: a TP collective whose world fits inside one node rides
    the intra fabric; once world exceeds gpus_per_node the collective
    crosses node boundaries and takes the inter fabric (slower → larger
    t_comm at the same per-rank bytes)."""

    @classmethod
    def setUpClass(cls):
        from backend.analyzer import fetch_config
        from tests.snapshots import shared_loader
        from backend.inference.framework_bindings import get_binding
        cls.resolved = shared_loader().match(fetch_config("meta-llama/Llama-2-7b-hf"), model_id="meta-llama/Llama-2-7b-hf")
        cls.binding = get_binding("vllm")

    def _rows(self, **pc_kwargs):
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        # H100-SXM-like split, per-direction unidirectional GB/s:
        # NVLink 4 intra (450 GB/s/dir), RDMA CX-7 inter (50 GB/s/dir),
        # 8 GPUs per node. Picking these as fixed numbers keeps the
        # test independent of the CSV (so a CSV update can't silently
        # break the assertion).
        res = analyze(
            self.resolved, parallel=ParallelConfig(**pc_kwargs),
            framework_binding=self.binding, batch_size=1, avg_input_tokens=4096, avg_output_tokens=0,
            peak_flops_per_sec=1e15, peak_hbm_bw_bytes_per_sec=3.35e12,
            intra_bw_bytes_per_sec=450e9,
            inter_bw_bytes_per_sec=50e9,
            gpus_per_node=8,
            intra_bw_label="NVLink",
            inter_bw_label="RDMA",
            breakdown=True,
        )
        return res.per_op_timing

    def test_tp8_takes_the_intra_nvlink_fabric(self):
        # TP=8 ≤ gpus_per_node=8 → single intra-only NVLink phase. The
        # bottleneck reduces to that phase, so the row-level label and
        # bandwidth match the intra fabric and t_comm = per_rank_bytes / intra_bw.
        rows = [r for r in self._rows(tp_attention=8, tp_dense_ffn=8)
                if r.get("comm_bytes_per_rank", 0) > 0
                and r["op"] == "all_reduce"]
        self.assertTrue(rows, "expected an all_reduce row at TP=8")
        for r in rows:
            self.assertEqual(r["comm_bw_label"], "NVLink")
            self.assertEqual(r["comm_bw_bytes_per_sec"], 450e9)
            # Single-phase intra-only case: t_comm == per_rank_bytes / intra_bw.
            self.assertAlmostEqual(
                r["t_comm_s"], r["comm_bytes_per_rank"] / 450e9,
                places=12,
            )
            # Single-phase rows collapse `comm_phases` to an empty list —
            # frontend uses that to skip the per-phase breakdown popover.
            self.assertEqual(r["comm_phases"], [])

    def test_tp16_crosses_nodes_uses_hierarchical_ar(self):
        # TP=16 > gpus_per_node=8 → cross-node all_reduce. NCCL /
        # Megatron run a 3-phase hierarchical schedule (intra-RS +
        # inter-AR on the shard + intra-AG) rather than a flat 16-rank
        # ring on the inter fabric; the row carries `comm_phases` with
        # the three sub-collectives and t_comm sums their per-phase
        # times. Catches the pre-fix `flat_ring(T) / inter_bw` model,
        # which materially under-counts intra-node hops.
        rows = [r for r in self._rows(tp_attention=16, tp_dense_ffn=16)
                if r.get("comm_bytes_per_rank", 0) > 0
                and r["op"] == "all_reduce"]
        self.assertTrue(rows, "expected an all_reduce row at TP=16")
        for r in rows:
            phases = r["comm_phases"]
            self.assertEqual(
                len(phases), 3,
                f"expected 3 hierarchical phases, got {phases}",
            )
            self.assertEqual(
                [p["name"] for p in phases],
                ["reduce_scatter", "all_reduce", "all_gather"],
            )
            self.assertEqual(
                [p["fabric"] for p in phases],
                ["intra", "inter", "intra"],
            )
            # Intra phases span one node (8 ranks); the inter phase
            # spans the cross-node domain (16/8 = 2 nodes).
            self.assertEqual([p["world"] for p in phases], [8, 2, 8])
            # Row total t_comm == Σ phase times — the whole point of
            # the hierarchical model.
            expected_t = sum(p["t_phase_s"] for p in phases)
            self.assertAlmostEqual(r["t_comm_s"], expected_t, places=12)
            # Bottleneck phase (largest t_phase_s) determines the
            # row-level summary bandwidth + label. Under realistic
            # H100 ratios (450 GB/s intra vs 50 GB/s inter) the inter
            # phase is the bottleneck → RDMA.
            bottleneck = max(phases, key=lambda p: p["t_phase_s"])
            self.assertEqual(r["comm_bw_label"], bottleneck["fabric_label"])
            self.assertEqual(
                r["comm_bw_bytes_per_sec"], bottleneck["bw_bytes_per_sec"],
            )
            self.assertEqual(r["comm_bw_label"], "RDMA")
            self.assertEqual(r["comm_bw_bytes_per_sec"], 50e9)

    def test_legacy_kwarg_still_works(self):
        # Pre-split callers used a single `interconnect_bw_bytes_per_sec`
        # kwarg. The alias must keep working: it mirrors to both intra
        # and inter so t_comm matches the legacy single-fabric behavior.
        from backend.inference.parallel_config import ParallelConfig
        from backend.inference.pipeline import analyze
        res = analyze(
            self.resolved, parallel=ParallelConfig(tp_attention=8),
            framework_binding=self.binding, batch_size=1, avg_input_tokens=4096, avg_output_tokens=0,
            peak_flops_per_sec=1e15, peak_hbm_bw_bytes_per_sec=3.35e12,
            interconnect_bw_bytes_per_sec=450e9,
            breakdown=True,
        )
        rows = [r for r in res.per_op_timing
                if r.get("comm_bytes_per_rank", 0) > 0]
        self.assertTrue(rows)
        for r in rows:
            self.assertGreater(r["t_comm_s"], 0)
            # Legacy alias mirrors to both fabrics → whichever one the
            # world size picks, the BW is the same 450e9.
            self.assertEqual(r["comm_bw_bytes_per_sec"], 450e9)

    def test_comm_world_one_short_circuits(self):
        # TP=1 → no comm rows at all (no peers). Defensive: even if
        # something synthesized a comm node with world=1, t_comm would
        # be zero and the row would carry empty bandwidth label.
        rows = self._rows(tp_attention=1, tp_dense_ffn=1)
        for r in rows:
            self.assertEqual(r.get("comm_bytes_per_rank", 0), 0)
            self.assertEqual(r.get("t_comm_s", 0), 0)
            self.assertEqual(r.get("comm_bw_label", ""), "")

    def test_per_op_rows_carry_shape_out(self):
        # Every leaf row in per_op_timing must surface its tensor shape
        # so the Per-Module Breakdown can render a Shape column. For
        # compute leaves this comes from the physical shape layer; for
        # materialized communication rows it comes from the typed payload
        # shape carried in the communication metadata layer.
        # Regression guard: a previous version forgot the comm fallback,
        # leaving AR rows with shape=None — the most identifying fact
        # about a collective was invisible.
        rows = self._rows(tp_attention=8, tp_dense_ffn=8)
        # Compute leaves: at least one linear must have a populated shape.
        compute_rows = [r for r in rows if r["op"] not in (
            "all_reduce", "reduce_scatter", "all_gather", "all_to_all",
            "permute", "allreduce_fused_rmsnorm",
        )]
        with_shape = [r for r in compute_rows if r.get("shape_out")]
        self.assertTrue(with_shape, "no compute leaf carries shape_out")
        for r in with_shape:
            self.assertIsInstance(r["shape_out"], list)
            self.assertTrue(all(isinstance(d, int) for d in r["shape_out"]))

        # Comm rows: AR rows in llama TP=8 must carry the producer's
        # shape (decompressed [B, S, H] for the FFN-exit reduce).
        comm_rows = [r for r in rows
                     if r["op"] == "all_reduce"
                     and r.get("comm_bytes_per_rank", 0) > 0]
        self.assertTrue(comm_rows, "expected at least one all_reduce row")
        for r in comm_rows:
            self.assertIsNotNone(
                r.get("shape_out"),
                f"all_reduce row {r['aggregation_id']!r} missing shape_out — "
                f"comm_meta tensor_shape fallback broken",
            )
            self.assertIsInstance(r["shape_out"], list)
            self.assertTrue(all(isinstance(d, int) for d in r["shape_out"]))

    def test_comm_bytes_formula_carries_symbolic_and_substituted(self):
        # The popover-grade formula is a {symbolic, substituted} dict so
        # the frontend can render the coefficient·T form alongside the
        # substituted numbers as first-class lines. `T` is EXPANDED to its
        # axis·dtype product — `2*(N-1)/N * (...·dt_a)` symbolic, with the
        # concrete dims·dtype on the substituted line — so the bytes are
        # traceable to the tensor shape, not an opaque T. `*` (not `·`) for
        # the coefficient so it's copy-paste safe back into Python / sympy.
        rows = [r for r in self._rows(tp_attention=8, tp_dense_ffn=8)
                if r.get("comm_bytes_per_rank", 0) > 0
                and r["op"] == "all_reduce"]
        self.assertTrue(rows)
        for r in rows:
            fmt = r["comm_bytes_formula"]
            self.assertIsInstance(fmt, dict)
            # coefficient kept; T expanded to a parenthesized axis·dt_a product.
            self.assertIn("2*(N-1)/N * (", fmt["symbolic"])
            self.assertIn("dt_a", fmt["symbolic"])
            self.assertIn("2*7/8", fmt["substituted"])
            self.assertEqual(r["comm_world"], 8)


if __name__ == "__main__":
    unittest.main()


class TestCollectiveLatency(unittest.TestCase):
    """A collective's fixed time is the link latency times the traversals
    its schedule waits on in sequence (`comm_volume.latency_hops`), added
    over the phases of a hierarchical schedule."""

    ALPHA = 2e-6

    def _latency(self, op, world, count=1, gpus_per_node=8, stride=1):
        return collective_latency(op=op, world=world, count_per_step=count,
                                  gpus_per_node=gpus_per_node,
                                  base_stride=stride, step_latency=self.ALPHA)

    def test_each_collective_waits_on_its_schedules_hops(self):
        from backend.arch.layout.comm_volume import latency_hops
        self.assertEqual(latency_hops("all_reduce", 8), 6)    # RS + AG, 3 rounds each
        self.assertEqual(latency_hops("reduce_scatter", 8), 3)
        self.assertEqual(latency_hops("all_gather", 5), 3)    # rounds round up
        self.assertEqual(latency_hops("all_to_all", 64), 6)   # Bruck
        self.assertEqual(latency_hops("send_recv", 2), 1)
        self.assertEqual(latency_hops("all_reduce", 1), 0)

    def test_a_hierarchical_all_reduce_adds_its_phases_hops(self):
        # 16 ranks over 2 nodes: intra RS (3) + inter AR of 2 (2) + intra AG (3).
        self.assertAlmostEqual(self._latency("all_reduce", 16), 8 * self.ALPHA)
        # Within a node: one flat all-reduce of 8.
        self.assertAlmostEqual(self._latency("all_reduce", 8), 6 * self.ALPHA)

    def test_every_instance_pays_it_and_bytes_do_not_move_it(self):
        self.assertAlmostEqual(self._latency("all_to_all", 16, count=58),
                               58 * 4 * self.ALPHA)
        t_small, _ = price_collective(
            op="all_reduce", world=8, tensor_bytes=1.0, count_per_step=1,
            fabric=Fabric(8, 4.5e11, 5e10, "NVLink", "RDMA", self.ALPHA),
            base_stride=1)
        self.assertAlmostEqual(t_small, 6 * self.ALPHA, delta=1e-10)


class TestTheFrameworkPicksTheAllReduce(unittest.TestCase):
    """Inside one NVLink domain the framework's dispatch (`collectives:` in
    its binding) picks the all-reduce algorithm by payload: one-shot (one
    traversal, a rank receives (N-1)T) below the threshold, two-shot (two,
    2(N-1)/N T) above it, NVLS where the switch reduces; NCCL's schedule
    past them, on a card without an NVLink domain, and across domains
    (`comm_cost.collective_ladder`)."""

    ALPHA = 2e-6

    def _rows(self, card, world, payload, framework="vllm", stride=1,
              architectures=()):
        from backend.inference.comm_cost import collective_ladder, ladder_segment
        from backend.inference.framework_bindings import get_binding
        from backend.inference.gpu_specs import get_gpu
        fabric = Fabric.of(get_gpu(card), self.ALPHA)
        from types import SimpleNamespace
        rungs = get_binding(framework).collectives_for(
            SimpleNamespace(architectures=architectures), None)["all_reduce"]
        seg = ladder_segment(collective_ladder(
            op="all_reduce", world=world, count_per_step=1, base_stride=stride,
            fabric=fabric, rungs=rungs), payload)
        return fabric, price_collective(
            op="all_reduce", world=world, tensor_bytes=payload,
            count_per_step=1, fabric=fabric, base_stride=stride,
            algorithm=seg.algorithm)

    def _one(self, *args, **kw):
        _fabric, (t, rows) = self._rows(*args, **kw)
        return t, [(r["algorithm"], r["name"], r["world"], r["hops"]) for r in rows]

    def test_a_tp8_decode_all_reduce_on_hgx_is_one_shot(self):
        # Kimi-K3 / DeepSeek-V3 at B=1: 7168 x 2 B. N T = 112 KiB <= 1 MiB.
        for card, bw in (("H100-SXM", 450e9), ("B300", 900e9)):
            t, rows = self._one(card, 8, 14336.0)
            self.assertEqual(rows, [("one_shot", "all_reduce", 8, 1)], card)
            # By hand: one traversal, and every rank receives 7 payloads.
            self.assertAlmostEqual(t, self.ALPHA + 7 * 14336 / bw, delta=1e-15)

    def test_above_the_one_shot_threshold_it_is_two_shot(self):
        # B=16: 229,376 B; N T = 1.75 MiB > 1 MiB.
        t, rows = self._one("H100-SXM", 8, 229376.0)
        self.assertEqual(rows, [("two_shot", "all_reduce", 8, 2)])
        self.assertAlmostEqual(t, 2 * self.ALPHA + 2 * 7 / 8 * 229376 / 450e9,
                               delta=1e-15)
        # The threshold is inclusive: exactly 1 MiB over the group is one-shot.
        self.assertEqual(self._one("H100-SXM", 8, 131072.0)[1][0][0], "one_shot")
        self.assertEqual(self._one("H100-SXM", 8, 131088.0)[1][0][0], "two_shot")

    def test_a_prefill_payload_is_nvls_on_hopper_and_two_shot_on_b300(self):
        T = 14680064.0   # 1024 tokens x 7168 x 2 B
        t, rows = self._one("H100-SXM", 8, T)
        self.assertEqual(rows, [("nvls", "all_reduce", 8, 2)])
        self.assertAlmostEqual(t, 2 * self.ALPHA + 7 / 8 * T / 450e9, delta=1e-12)
        self.assertEqual(self._one("B300", 8, T)[1],
                         [("two_shot", "all_reduce", 8, 2)])

    def test_without_an_nvlink_domain_it_is_nccls_schedule(self):
        # PCIe cards: no peer-to-peer NVLink, no switch — every payload runs
        # NCCL's schedule, its log-depth hops and the ring's bytes.
        for card, world in (("H100-PCIe", 2), ("H100-PCIe", 4),
                            ("RTX-4090", 2), ("RTX-4090", 8)):
            for payload in (14336.0, 14680064.0):
                _t, rows = self._one(card, world, payload)
                self.assertEqual(rows, [("nccl", "all_reduce", world,
                                         2 * (world - 1).bit_length())],
                                 (card, world, payload))

    def test_nvlink_without_a_reducing_switch_skips_nvls(self):
        # A100's NVSwitch does not reduce in the network: one-shot and
        # two-shot run, a payload past them is NCCL's.
        self.assertEqual(self._one("A100-80G-SXM", 8, 14336.0)[1][0][0], "one_shot")
        self.assertEqual(self._one("A100-80G-SXM", 8, 14680064.0)[1][0][0], "nccl")

    def test_two_shot_and_nvls_need_their_group_sizes(self):
        # NVLS at 4, 6 and 8 ranks; a 2-rank group past two-shot is NCCL's,
        # and so is a 3-rank one, which no vLLM kernel takes at all.
        self.assertEqual(self._one("H100-SXM", 2, 14680064.0)[1][0][0], "nccl")
        self.assertEqual(self._one("H100-SXM", 4, 14680064.0)[1][0][0], "nvls")
        self.assertEqual(self._one("H100-SXM", 3, 7168.0)[1][0][0], "nccl")
        self.assertEqual(self._one("H100-SXM", 3, 7168.0, framework="sglang")[1][0][0],
                         "one_shot")

    def test_across_hgx_nodes_the_hierarchy_stays(self):
        # TP16 on 8-GPU HGX nodes: NCCL's three phases, each its own hops.
        _t, rows = self._one("H100-SXM", 16, 14336.0)
        self.assertEqual(rows, [("nccl", "reduce_scatter", 8, 3),
                                ("nccl", "all_reduce", 2, 2),
                                ("nccl", "all_gather", 8, 3)])
        # An outer group strided across nodes is one member per node.
        _t, rows = self._one("H100-SXM", 4, 14336.0, stride=8)
        self.assertEqual(rows, [("nccl", "all_reduce", 4, 4)])

    def test_an_nvl72_rack_runs_the_group_sizes_the_kernels_take(self):
        # GB200 NVL72: TP16 stays inside the rack's NVLink domain, and
        # FlashInfer has a 16-rank table on SM100: one-shot, then two-shot.
        self.assertEqual(self._one("GB200", 16, 7168.0)[1],
                         [("one_shot", "all_reduce", 16, 1)])
        self.assertEqual(self._one("GB200", 16, 524288.0)[1],
                         [("two_shot", "all_reduce", 16, 2)])
        # Symmetric memory's multimem has no 16-rank entry: NCCL.
        self.assertEqual(self._one("GB200", 16, 14680064.0)[1][0][0], "nccl")
        # 32 and 72 ranks are one domain too, but no framework kernel takes
        # them: NCCL's schedule, flat inside the rack.
        for world in (32, 72):
            for payload in (7168.0, 14680064.0):
                self.assertEqual(self._one("GB200", world, payload)[1],
                                 [("nccl", "all_reduce", world,
                                   2 * (world - 1).bit_length())],
                                 (world, payload))
        self.assertEqual(self._one("GB200", 32, 7168.0, framework="sglang")[1][0][0],
                         "nccl")

    def test_sglang_reads_its_own_dispatch(self):
        _t, rows = self._one("H100-SXM", 8, 14336.0, framework="sglang")
        self.assertEqual(rows, [("one_shot", "all_reduce", 8, 1)])
        # SM100 keeps two-shot to 8 MiB, SM90 to 512 KiB.
        self.assertEqual(self._one("B200", 8, 4194304.0, framework="sglang")[1][0][0],
                         "two_shot")
        self.assertEqual(self._one("H100-SXM", 8, 4194304.0, framework="sglang")[1][0][0],
                         "nvls")

    def test_a_fabric_with_no_card_runs_nccl(self):
        # Bandwidths alone admit no framework kernel: the textbook schedule.
        from backend.inference.comm_cost import CollectiveRung, collective_ladder
        ladder = collective_ladder(
            op="all_reduce", world=8, count_per_step=1, base_stride=1,
            fabric=Fabric(8, 4.5e11, 5e10, "NVLink", "RDMA", self.ALPHA),
            rungs=(CollectiveRung("one_shot", max_bytes_times_world=1 << 20,
                                  worlds=(8,)),))
        self.assertEqual([s.algorithm for s in ladder], ["nccl", "nccl"])
        self.assertEqual(ladder[0].max_bytes, 0.0)


class TestBlackwellAndFusionArchitectures(TestTheFrameworkPicksTheAllReduce):
    """NVLS by the capability's group sizes, and sglang's FlashInfer path for
    the model classes it turns AR fusion on for."""

    T = 14680064.0   # a 1024-token prefill all-reduce, 7168 x 2 B

    def test_a_blackwell_tp4_prefill_all_reduce_is_two_shot(self):
        # vLLM: no multimem at 4 ranks on SM100; FlashInfer's two-shot to
        # 32 MiB (64 MiB on SM103). sglang: no multicast at 4 ranks on SM100.
        for card in ("B200", "B300"):
            for fw in ("vllm", "sglang"):
                t, rows = self._one(card, 4, self.T, framework=fw)
                self.assertEqual(rows, [("two_shot", "all_reduce", 4, 2)],
                                 (card, fw))
                # By hand: two traversals, 2 x 3/4 of the payload.
                self.assertAlmostEqual(
                    t, 2 * self.ALPHA + 1.5 * self.T / 900e9, delta=1e-12)
        # Past FlashInfer's SM100 32 MiB cap, vLLM runs NCCL.
        self.assertEqual(self._one("B200", 4, 64e6)[1][0][0], "nccl")
        # Hopper keeps NVLS at 4 ranks.
        self.assertEqual(self._one("H100-SXM", 4, self.T)[1][0][0], "nvls")

    def test_sglang_runs_flashinfer_two_shot_for_its_fusion_architectures(self):
        dsv3 = ("DeepseekV3ForCausalLM",)
        self.assertEqual(
            self._one("B200", 8, self.T, framework="sglang",
                      architectures=dsv3)[1],
            [("two_shot", "all_reduce", 8, 2)])
        # Below the switch it is still one-shot.
        self.assertEqual(
            self._one("B200", 8, 14336.0, framework="sglang",
                      architectures=dsv3)[1][0][0], "one_shot")
        # Another model class runs the v2 kernel: multicast at 8 ranks.
        self.assertEqual(
            self._one("B200", 8, self.T, framework="sglang",
                      architectures=("LlamaForCausalLM",))[1][0][0], "nvls")
        # Hopper: the fusion rung is SM100's only.
        self.assertEqual(
            self._one("H100-SXM", 8, self.T, framework="sglang",
                      architectures=dsv3)[1][0][0], "nvls")


class TestCardFabricHasOneStatement(unittest.TestCase):
    """A card's fabric is `Fabric.of` and nothing else.

    Three pricing roads used to build one from six loose preparation fields
    plus a `topology` splat (`capacity._point_rows`,
    `_stamp_gpu_on_result`, `_chunk_point_rows_and_phase`); they now ask
    `_CapacityPreparation.fabric`. This pins the two readings equal on
    every row of the sheet, so the consolidation is checked against the
    form it replaced rather than against itself.
    """

    def test_of_equals_the_loose_field_assembly(self):
        from backend.inference import gpu_specs as gs
        from backend.inference.comm_cost import topology
        latency = 1.5e-6
        for gpu in gs.load_gpu_specs():
            was = Fabric(
                gs.gpus_per_node(gpu),
                gs.peak_intra_bw_bytes_per_sec(gpu),
                gs.peak_inter_bw_bytes_per_sec(gpu),
                gs.intra_bw_label(gpu), gs.inter_bw_label(gpu),
                latency,
                scaleup_domain=gs.ranks_per_scaleup_domain(gpu),
                )._replace(**topology(gpu))
            self.assertEqual(Fabric.of(gpu, latency), was, gpu.sku)


class TheReachOfAGroup(unittest.TestCase):
    """A collective's group spans one chassis, a scale-up domain, or more
    than one — and a measurement of each must answer only for its own.

    Before this, a sheet row meant one chassis and a group either fitted it
    or did not. A supernode ends that: on an Atlas A3, 32 ranks leave the
    16-die chassis and are still on the scale-up fabric, a domain away in
    bandwidth from the RoCE the two-valued reading prices them at.
    """

    def test_a_group_inside_one_chassis_is_host(self):
        for world in (2, 8, 16):
            self.assertEqual(
                comm_reach(world, base_stride=1, ranks_per_host=16),
                COMM_REACH_HOST, world)

    def test_without_a_wider_domain_leaving_the_chassis_is_scaleout(self):
        """What the two-valued reading has always said, unchanged: a card
        declaring no domain past its chassis has nothing in between."""
        self.assertEqual(
            comm_reach(32, base_stride=1, ranks_per_host=16),
            COMM_REACH_SCALEOUT)

    def test_a_declared_supernode_puts_a_tier_in_between(self):
        def reach(n):
            return comm_reach(n, base_stride=1, ranks_per_host=16,
                              ranks_per_scaleup_domain=384)
        self.assertEqual(reach(16), COMM_REACH_HOST)
        self.assertEqual(reach(32), COMM_REACH_SCALEUP)
        self.assertEqual(reach(384), COMM_REACH_SCALEUP)
        self.assertEqual(reach(385), COMM_REACH_SCALEOUT)

    def test_stride_spreads_a_small_group_past_its_chassis(self):
        """An outer axis composed over an inner one is spaced out: four
        ranks at stride 8 touch four chassis of 16, not one."""
        self.assertEqual(
            comm_reach(4, base_stride=8, ranks_per_host=16,
                       ranks_per_scaleup_domain=384),
            COMM_REACH_SCALEUP)

    def test_the_price_and_the_key_divide_the_same_way(self):
        """`collective_phases` and `comm_reach` must not disagree about
        where a group sits: one says intra, the other says host, and they
        are the same sentence. Both read `members_within`."""
        for world, stride, g in ((8, 1, 16), (32, 1, 16), (4, 8, 16),
                                 (16, 1, 8), (12, 1, 16), (2, 1, 16)):
            phases = collective_phases("all_reduce", world, 1 << 20,
                                       gpus_per_node=g, base_stride=stride)
            priced_all_intra = bool(phases) and all(
                p["fabric"] == "intra" for p in phases)
            keyed_host = comm_reach(world, base_stride=stride,
                                    ranks_per_host=g) == COMM_REACH_HOST
            self.assertEqual(priced_all_intra, keyed_host,
                             f"world={world} stride={stride} g={g}")


class WhatACollectiveCellIsKeyedOn(unittest.TestCase):
    """The reach belongs in the key, and belongs there by its ABSENCE for
    the common case."""

    HOST = '{"n_comm_elems":4194304,"n_comm_ranks":8,"n_comm_stride":1}'

    @staticmethod
    def _sig(name="all_reduce"):
        import backend.inference.op_signatures_catalog  # noqa: F401
        from backend.inference.op_signatures import get_signature
        return get_signature(name)

    def _keyed(self, reach, name="all_reduce"):
        import json
        return self._sig(name).project(json.dumps(
            {**json.loads(self.HOST), COMM_DOMAIN_AXIS: int(reach)},
            separators=(",", ":"), sort_keys=True))

    def test_an_intra_host_row_keys_exactly_as_it_did(self):
        """Every collective row measured so far is intra-host, so this is
        the whole backward-compatibility claim: the axis costs them
        nothing and no corpus row is rewritten to gain it."""
        self.assertEqual(self._sig().project(self.HOST), self.HOST)

    def test_the_same_payload_and_group_at_two_reaches_are_two_keys(self):
        self.assertNotEqual(self._keyed(COMM_REACH_SCALEUP),
                            self._sig().project(self.HOST))

    def test_scaleup_and_scaleout_do_not_share_a_key_either(self):
        """The middle tier is the point: a supernode measurement must not
        answer for a group that left the supernode."""
        self.assertNotEqual(self._keyed(COMM_REACH_SCALEUP),
                            self._keyed(COMM_REACH_SCALEOUT))

    def test_every_collective_signature_declares_the_axis(self):
        """A signature that forgot it would go on sharing one key across
        fabrics, which is the defect this axis exists to end."""
        for name in ("all_reduce", "all_gather", "reduce_scatter",
                     "all_to_all", "send_recv"):
            self.assertIn(COMM_DOMAIN_AXIS, self._sig(name).variant_dims, name)

    def test_the_axis_is_not_a_cell_dim(self):
        """It identifies by presence. In `cell_dims` it would be REQUIRED,
        and every existing row — none of which carries it — would stop
        keying at all."""
        for name in ("all_reduce", "send_recv"):
            self.assertNotIn(COMM_DOMAIN_AXIS, self._sig(name).cell_dims, name)

    def test_a_crossing_lookup_misses_a_corpus_of_host_rows(self):
        """The property the axis exists for, asserted on the real lookup
        and not on two strings being unequal.

        The corpus holds intra-host collective rows and nothing else. A
        deployment whose group leaves the chassis must come away with NO
        measurement — falling back to the roofline — rather than being
        handed the intra-host row because its key happens to omit an axis
        the asking key carries. An absent axis is the DEFAULT (host), never
        a wildcard that matches every reach, and the difference between
        those two readings is a cross-fabric collective priced at HCCS.
        """
        import csv
        from backend.inference.op_calibration import (
            CalibrationRegistry, BenchmarkKey, merge_validated_rows,
            physical_time_for)
        path = PROJECT_ROOT / "tests/golden/ascend_vllm_ascend.csv"
        with open(path, newline="") as handle:
            rows = [r for r in csv.DictReader(handle) if r["op"] in COMM_OPS]
        registry = merge_validated_rows(
            CalibrationRegistry(), [dict(r) for r in rows],
            source="<test>", origin="shared")

        host_keys = [BenchmarkKey.from_csv_row(r).to_physical() for r in rows]

        def timing(key):
            return physical_time_for(
                registry, key,
                current_peak_tflops_at=lambda *_a, **_k: 248.4,
                current_peak_bw_gbps=1600.0)

        # The corpus answers for the reach it was taken at ...
        hit = [k for k in host_keys if timing(k) is not None]
        self.assertTrue(hit, "no intra-host collective row answered at all, "
                             "so a miss below would prove nothing")

        # ... and for no other.
        import json
        for reach in (COMM_REACH_SCALEUP, COMM_REACH_SCALEOUT):
            for key in host_keys:
                crossing = replace(key, shape_json=json.dumps(
                    {**json.loads(key.shape_json), COMM_DOMAIN_AXIS: int(reach)},
                    separators=(",", ":"), sort_keys=True))
                self.assertEqual(
                    registry.get_physical(crossing), [],
                    f"{reach} key found an intra-host row")
                self.assertIsNone(
                    timing(crossing),
                    f"{reach} group was handed an intra-host measurement")

    def test_the_measured_corpus_still_keys_where_it_did(self):
        """The 12 Ascend collective rows were all taken on one chassis
        (the sheet row's note says every measurement on it is intra-node),
        so not one of them may gain the axis."""
        import csv
        from backend.inference.op_calibration import BenchmarkKey
        path = PROJECT_ROOT / "tests/golden/ascend_vllm_ascend.csv"
        with open(path, newline="") as handle:
            rows = [r for r in csv.DictReader(handle)
                    if r["op"] in COMM_OPS]
        self.assertTrue(rows, "no collective rows to check")
        for row in rows:
            shape = BenchmarkKey.from_csv_row(row).to_physical().shape_json
            self.assertNotIn(COMM_DOMAIN_AXIS, shape, row["op"])


if __name__ == "__main__":
    unittest.main()


class ASupernodeIsScaleUpAcrossMachines(unittest.TestCase):
    """Both halves of the question, on the sheet row that answers it.

    `gpus_per_node` is the CHASSIS and `ranks_per_scaleup_domain` is the
    fast fabric's reach. They coincide on every card that has no
    supernode, and on an Atlas A3 they do not: 16 against 768. Getting
    that wrong is not a crash, it is 32 ranks priced on a 25 GB/s NIC
    that carried none of their traffic.
    """

    def _ascend(self):
        from backend.inference.gpu_specs import get_gpu
        gpu = get_gpu("Ascend-910C-A3")
        self.assertIsNotNone(gpu, "the Ascend row is what this tests")
        return gpu

    def test_the_sheet_separates_the_chassis_from_the_domain(self):
        from backend.inference.gpu_specs import (gpus_per_node,
                                                 ranks_per_scaleup_domain)
        gpu = self._ascend()
        self.assertEqual(gpus_per_node(gpu), 16)
        self.assertEqual(ranks_per_scaleup_domain(gpu), 768)

    def test_the_lookup_tiers_a_crossing_group_as_scaleup(self):
        """A measured row keyed `scaleup` is only found if the estimate
        asks for `scaleup`."""
        from backend.inference.gpu_specs import (gpus_per_node,
                                                 ranks_per_scaleup_domain)
        gpu = self._ascend()
        host, domain = gpus_per_node(gpu), ranks_per_scaleup_domain(gpu)
        for world, want in ((8, COMM_REACH_HOST), (16, COMM_REACH_HOST),
                            (32, COMM_REACH_SCALEUP),
                            (64, COMM_REACH_SCALEUP),
                            (768, COMM_REACH_SCALEUP),
                            (769, COMM_REACH_SCALEOUT),
                            (1024, COMM_REACH_SCALEOUT)):
            self.assertEqual(
                comm_reach(world, base_stride=1, ranks_per_host=host,
                           ranks_per_scaleup_domain=domain),
                want, f"world={world}")

    def test_a_world_with_no_measurement_is_priced_on_the_fast_fabric(self):
        """The half a reach code cannot cover.

        64 and 96 ranks are not in the corpus and will be priced from the
        roofline. Inside the supernode that traffic never touches a NIC,
        so pricing it on `rdma_gbps` would be wrong by the better part of
        an order of magnitude -- and silently, since a number comes back
        either way.
        """
        gpu = self._ascend()
        fabric = Fabric.of(gpu, 2e-6)
        for world in (16, 32, 64, 96, 768):
            _t, phases = price_collective(
                op="all_reduce", world=world, tensor_bytes=1 << 23,
                count_per_step=1, fabric=fabric, base_stride=1)
            labels = {p["fabric_label"] for p in phases}
            self.assertEqual(labels, {"HCCS"}, f"world={world} left the "
                                               f"scale-up fabric: {labels}")

    def test_past_the_domain_it_is_the_nic(self):
        gpu = self._ascend()
        _t, phases = price_collective(
            op="all_reduce", world=1024, tensor_bytes=1 << 23,
            count_per_step=1, fabric=Fabric.of(gpu, 2e-6), base_stride=1)
        self.assertIn("RDMA", {p["fabric_label"] for p in phases})

    def test_a_card_with_no_supernode_is_unchanged(self):
        """Every other row on the sheet leaves the column empty, and must
        read exactly as it did: this is what keeps the GPU baseline
        byte-identical."""
        from backend.inference import gpu_specs as gs
        for gpu in gs.load_gpu_specs():
            if gpu.sku == "Ascend-910C-A3":
                continue
            self.assertEqual(
                gs.ranks_per_scaleup_domain(gpu), gs.gpus_per_node(gpu),
                f"{gpu.sku} gained a domain it never declared")
