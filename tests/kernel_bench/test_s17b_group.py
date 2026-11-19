"""S17b Commit B: group mapping, watchdog, unique writer, max_r latency.

These pin the process model without HCCL: stub children return rows, the
parent aggregates, and a dead or silent rank kills the rest. Die mapping
stays in `group.py` and never enters the 12 identity columns.
"""
from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


_JOB = {
    "op": "all_reduce",
    "signature": "all_reduce",
    "dims": {"n_comm_elems": 8, "n_comm_ranks": 2, "n_comm_stride": 1},
    "dtype_env": {"activation": "bf16"},
    "phase": "prefill",
}


def _ok_rank(cfg, queue):
    rank = int(cfg["rank"])
    queue.put({
        "rank": rank,
        "rows": [{
            "kernel_impl": "hccl_all_reduce",
            "p50_us": f"{10.0 + rank:.4f}",
            "p95_us": f"{12.0 + rank:.4f}",
            "latency_us": f"{10.0 + rank:.4f}",
            "outcome": "measured",
            "launch_signature": "dist.all_reduce()",
        }],
    })


def _hang_rank(cfg, queue):
    time.sleep(60)


def _exit_rank(cfg, queue):
    if int(cfg["rank"]) == 1:
        os._exit(1)
    time.sleep(60)


def _error_rank(cfg, queue):
    if int(cfg["rank"]) == 1:
        queue.put({"rank": 1, "rows": [], "error": "rank 1: boom"})
        return
    _ok_rank(cfg, queue)


def _env():
    from kernel_bench.env import Env
    return Env("2.11", None, None, None, None)


#: What a child reported back about the config it was handed. A module-level
#: queue is not available to a spawned process, so the rank puts the fields
#: it cares about on the queue the parent already reads.
def _cfg_echo_rank(cfg, queue):
    rank = int(cfg["rank"])
    queue.put({
        "rank": rank,
        "rows": [{
            "kernel_impl": "hccl_all_reduce",
            "p50_us": f"{10.0 + rank:.4f}",
            "p95_us": f"{12.0 + rank:.4f}",
            "latency_us": f"{10.0 + rank:.4f}",
            "outcome": "measured",
            "launch_signature": "dist.all_reduce()",
            # Not a real column: the test reads it off the aggregated row to
            # see what the parent put in `cfg`.
            "_saw_peak_bw": repr(cfg.get("peak_bw_gbps")),
            "_saw_gpu_sku": repr(cfg.get("gpu_sku")),
        }],
    })


class TestDieMappingStaysOutOfIdentity(unittest.TestCase):
    def test_die_id_is_rank_times_stride(self):
        from kernel_bench.group import dies_for
        self.assertEqual(dies_for(4, stride=1), [0, 1, 2, 3])
        self.assertEqual(dies_for(4, stride=2), [0, 2, 4, 6])
        self.assertEqual(dies_for(2, stride=2, origin=1), [1, 3])

    def test_a_mixed_job_list_unions_the_dies(self):
        from kernel_bench.group import dies_for_jobs
        local = {"op": "linear", "dims": {"B": 1}}
        comm = dict(_JOB)
        self.assertIsNone(dies_for_jobs([local]))
        self.assertEqual(dies_for_jobs([comm], stride=2), [0, 2])
        self.assertEqual(dies_for_jobs([local, comm], stride=2), [0, 2])


class TestMaxAcrossRanksIsTheLatency(unittest.TestCase):
    def test_the_row_is_the_slowest_rank(self):
        from kernel_bench.group import aggregate_latency, spread_note
        rows = [
            {"_rank": 0, "kernel_impl": "hccl_all_reduce", "p50_us": "10.0",
             "p95_us": "11.0", "latency_us": "10.0", "outcome": "measured",
             "launch_signature": "dist.all_reduce()"},
            {"_rank": 1, "kernel_impl": "hccl_all_reduce", "p50_us": "13.0",
             "p95_us": "14.0", "latency_us": "13.0", "outcome": "measured",
             "launch_signature": "dist.all_reduce()"},
        ]
        out = aggregate_latency(rows)
        self.assertEqual(out["p50_us"], "13.0000")
        self.assertEqual(out["latency_us"], "13.0000")
        self.assertEqual(out["p95_us"], "14.0000")
        self.assertIn("max/min=", out["_rank_spread"])
        self.assertIn("rank_spread", out["launch_signature"])
        self.assertNotIn("_rank", out)
        self.assertIn("max/min=1.3", spread_note([10.0, 13.0]))

    def test_one_parent_row_from_two_children(self):
        from kernel_bench.group import measure_job
        rows = measure_job(_JOB, env=_env(), take_lock=False, timeout_s=15,
                           worker=_ok_rank)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["p50_us"], "11.0000")
        self.assertEqual(rows[0]["kernel_impl"], "hccl_all_reduce")


class TestTheGroupDiesTogether(unittest.TestCase):
    def test_a_silent_rank_kills_the_group(self):
        from kernel_bench.group import GroupFailed, measure_job
        t0 = time.monotonic()
        with self.assertRaises(GroupFailed) as caught:
            measure_job(_JOB, env=_env(), take_lock=False, timeout_s=1.5,
                        worker=_hang_rank)
        self.assertIn("timed out", str(caught.exception))
        self.assertLess(time.monotonic() - t0, 8.0)

    def test_an_exited_rank_kills_the_group(self):
        from kernel_bench.group import GroupFailed, measure_job
        with self.assertRaises(GroupFailed) as caught:
            measure_job(_JOB, env=_env(), take_lock=False, timeout_s=8,
                        worker=_exit_rank)
        self.assertIn("exited", str(caught.exception))

    def test_a_reported_error_kills_the_group(self):
        from kernel_bench.group import GroupFailed, measure_job
        with self.assertRaises(GroupFailed) as caught:
            measure_job(_JOB, env=_env(), take_lock=False, timeout_s=15,
                        worker=_error_rank)
        self.assertIn("boom", str(caught.exception))


class TestTheParentIsTheOnlyWriter(unittest.TestCase):
    def test_the_child_never_opens_the_csv(self):
        source = (PROJECT_ROOT / "kernel_bench" / "rankworker.py").read_text()
        self.assertNotIn("from .emit", source)
        self.assertNotIn("IncrementalCsvWriter(", source)
        self.assertNotIn("results.csv", source)


class TestHcclIsAGroupKernel(unittest.TestCase):
    def test_the_four_collectives_are_enabled_and_named(self):
        from kernel_bench.providers import all_providers
        hccl = [p for p in all_providers()
                if p.kernel_impl.startswith("hccl_")]
        self.assertEqual(sorted(p.signature for p in hccl),
                         ["all_gather", "all_reduce", "all_to_all",
                          "reduce_scatter"])
        for p in hccl:
            self.assertEqual(p.world, "n_comm_ranks")
            self.assertFalse(p.disabled_reason)

    def test_the_child_is_handed_the_card_the_parent_resolved(self):
        """The regression: a group measured correctly and wrote nothing.

        `emit.py:507` writes `peak_bw_at_test` from `env.peak_bw_gbps`, and
        an empty one fails schema validation, so every row a group measured
        was dropped at persist. The parent resolves the card off the BOX
        when the driver's device name names no sheet SKU (an Ascend die
        reports `Ascend910_9362`, whose family token carries no digit), so a
        child that only re-detects gets `None` and writes "".

        Both facts travel, because the SKU alone does not carry the
        bandwidth: `cli._cmd_run` looks the card up and replaces BOTH.
        """
        from dataclasses import replace

        from kernel_bench.group import measure_job
        env = replace(_env(), gpu_sku="Ascend-910C-A3", peak_bw_gbps=1600.0)
        rows = measure_job(_JOB, env=env, take_lock=False, timeout_s=15,
                           worker=_cfg_echo_rank)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["_saw_peak_bw"], repr(1600.0))
        self.assertEqual(rows[0]["_saw_gpu_sku"], repr("Ascend-910C-A3"))

    def test_the_empty_bandwidth_column_is_what_schema_rejects(self):
        """The other half of the failure: why an empty column loses the row.

        The group measured correctly and the log said `measured 2 rows`;
        both were then dropped at persist. That step is a REQUIRED column
        check, so a child that inherits no bandwidth does not fail loudly —
        it writes "" and the row disappears. Asserted against the validator
        itself rather than a copy of its list.
        """
        from backend.inference.benchmarks import MEASUREMENT_COLUMNS
        self.assertIn("peak_bw_at_test", MEASUREMENT_COLUMNS)

    def test_a_single_rank_is_not_a_group(self):
        from kernel_bench.invocation import Invocation
        from kernel_bench.providers import all_providers
        p = next(x for x in all_providers()
                 if x.kernel_impl == "hccl_all_reduce")
        inv = Invocation(
            signature="all_reduce", op="all_reduce",
            dims={"n_comm_elems": 8, "n_comm_ranks": 1, "n_comm_stride": 1},
            dtype_env={"activation": "bf16"}, phase="prefill", tp=1)
        reasons = list(p.refusals(inv, _env()))
        self.assertTrue(any(r.startswith("world:") for r in reasons), reasons)
