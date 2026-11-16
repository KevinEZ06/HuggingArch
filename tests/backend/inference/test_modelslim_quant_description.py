"""What a checkpoint quantised is the checkpoint's to say, not a preset's.

GLM-5.3-Flash ships twice. The official release is block-FP8 across
attention and FFN; an msModelSlim export for Ascend quantised the FFN to
INT8 and left everything else BF16. Their `config.json` files are
byte-identical, so nothing in the config tells them apart — the difference
is which tensors carry a quantised format, which `quant_context` is the
one place to say.

These assert that `quant_context.per_role` can express that export, by
building it and walking it. The export is NOT registered as a model id of
its own: its tensors are also NAMED differently (`attn_hc.base` where the
official checkpoint writes `hc_attn_base`, and 349 header entries the
official spec does not declare), so a spec for it is a spec written
against its own names rather than this one with two lines changed. That
is its own piece of work; what is settled here is the quantisation
question, which needed no engine change at all.

Read off the real export at /data3/GLM-5.3-Flash-w8a8 (82 shards,
113,353 tensors, headers read directly):

  * 111,843 `W8A8_DYNAMIC` entries, every one an expert's gate/up/down;
  * 27 more in the first three layers — the dense FFNs that
    `first_k_dense_replace: 3` leaves before the MoE stack starts;
  * every other tensor `FLOAT`, including all 957 attention ones.

Of the 100 distinct tensor-path shapes in its description file, NONE
carries both labels: the quantisation boundary falls on roles, which is
why two `per_role` rows are the whole of it.
"""
from __future__ import annotations

import collections
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.arch.layout.parallel_config import ParallelConfig  # noqa: E402
from backend.arch.loader import SpecLoader  # noqa: E402
from tests.snapshots import snapshot_config  # noqa: E402
from tests.sweep_parity import Deployment  # noqa: E402

MODEL = "zai-org/GLM-5.3-Flash"
PARALLEL = dict(tp_attention=8, tp_dense_ffn=8, ep_moe=8)


def _resolved():
    """A fresh resolve, so an injected `quant_context` cannot leak into
    another test through the shared loader's cache."""
    return SpecLoader().match(snapshot_config(MODEL), model_id=MODEL)


def _as_the_export_quantised_it(resolved):
    """The export's topology on this spec: the FFN roles INT8, the rest
    left at the width the checkpoint stores."""
    quant = resolved.spec.quant_context
    entry = type(next(iter(quant.per_role.values())))
    w8a8 = entry(method="modelslim-w8a8-dynamic")
    quant.per_role = {**{role: entry(method="none") for role in quant.per_role},
                      "ffn_dense": w8a8, "ffn_moe": w8a8}
    return resolved


def _estimate(resolved=None, preset=None):
    knobs = {"framework": "vllm_ascend"}
    if preset:
        knobs["precision"] = {"preset": preset}
    if resolved is not None:
        knobs["resolved"] = resolved
    return Deployment(model_id=MODEL, gpu_sku="Ascend-910C-A3", knobs=knobs,
                      parallel=ParallelConfig(**PARALLEL),
                      ).direct(avg_input_tokens=2048, prefill_batch=1,
                               decode_batch=16)


def _weight_widths(result):
    out = collections.defaultdict(collections.Counter)
    for row in result.get("per_op_timing") or []:
        if row.get("weight_dtype"):
            out[row.get("role") or "?"][row["weight_dtype"]] += 1
    return out


def _casts(result):
    return sum(1 for row in (result.get("per_op_timing") or [])
               if row.get("op") == "quantise_activation")


class TestPerRoleCanExpressTheExport(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.export = _estimate(_as_the_export_quantised_it(_resolved()))
        cls.widths = _weight_widths(cls.export)

    def test_both_ffn_roles_turn_int8(self):
        for role in ("ffn_moe", "ffn_dense"):
            self.assertEqual(set(self.widths[role]), {"INT8"},
                             f"{role}: {dict(self.widths[role])}")

    def test_the_attention_stack_stays_as_stored(self):
        for role in ("attention", "attention_linear", "attention_kv_up",
                     "attention_indexer"):
            widths = set(self.widths[role])
            self.assertNotIn("INT8", widths, f"{role}: {widths}")


class TestItIsNeitherNeighbour(unittest.TestCase):
    """The two answers this one is not."""

    def test_the_official_release_quantises_the_ffn_to_fp8(self):
        official = _weight_widths(_estimate())
        self.assertEqual(set(official["ffn_moe"]), {"F8_E4M3"})
        export = _weight_widths(
            _estimate(_as_the_export_quantised_it(_resolved())))
        self.assertEqual(set(export["ffn_moe"]), {"INT8"})

    def test_the_preset_quantises_attention_the_export_did_not(self):
        """The preset answers "what would full W8A8 cost", which is a
        different question and a different answer: it turns the attention
        linears INT8 and grafts a cast in front of each, where this export
        runs them BF16 with none."""
        preset = _estimate(preset="w_int8_w8a8")
        export = _estimate(_as_the_export_quantised_it(_resolved()))
        self.assertIn("INT8", _weight_widths(preset)["attention_linear"])
        self.assertNotIn("INT8", _weight_widths(export)["attention_linear"])

    def test_the_preset_grafts_casts_the_export_does_not_need(self):
        # The count is the visible cost of the difference: a cast per
        # quantised input, and the export quantises far fewer inputs.
        preset = _casts(_estimate(preset="w_int8_w8a8"))
        export = _casts(_estimate(_as_the_export_quantised_it(_resolved())))
        self.assertGreater(preset, export + 10, f"{preset} vs {export}")


if __name__ == "__main__":
    unittest.main()
