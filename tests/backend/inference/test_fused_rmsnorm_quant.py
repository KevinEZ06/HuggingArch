"""A norm that writes a quantised GEMM's input, and the DSA query launch.

Two launches the serving graph used to bill as two:

* `fused_rmsnorm_quant` — an RMSNorm with no residual add whose one reader
  is the `quantise_activation` cast of a quantised GEMM (MLA's
  `q_a_layernorm` → `q_b_proj`). vLLM's `RMSNormQuantFusionPass` rewrites
  the pair into one launch; the plan is bound for sglang too, as
  `fused_add_rmsnorm`'s optional cast is (the graph models the better
  form). It ties on required nodes with `fused_add_rmsnorm` and
  `fused_q_kv_rmsnorm`, and loses to both on all nodes (`match_order`), so
  an add's chain stays the add's launch and a q/kv a-norm pair stays whole.
* `fused_dsa_q_rope_quant` — vLLM's `fused_q`: the main q_pe rotation and
  the DSA indexer query's rotation + FP8 quant in one launch, for
  DeepSeek-V3.2 (NeoX indexer rope) and GLM-5 (interleaved) alike.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.snapshots import resolve_snapshot

from backend.arch.precision_profile import PRESETS, PrecisionProfile
from backend.inference.framework_bindings import ServingDeployment, get_binding
from backend.inference.parallel_config import ParallelConfig
from backend.inference.serving import rewrite_serving_graph
from backend.inference.step_eval import evaluate_step


#: The plans these tests read the member sets of.
_PLANS = ("fused_rmsnorm_quant", "fused_add_rmsnorm", "fused_q_kv_rmsnorm",
          "fused_dsa_q_rope_quant")

#: A deployment that keeps the MLA cache wide and the index key as shipped
#: (FP8): vLLM's `fused_q` then takes its BF16-query path (`_fp8_query`
#: false, deepseek_v32/attention.py:238-239, :446), the path
#: `fused_dsa_q_rope_quant` prices. With an FP8 latent the launch would also
#: quantise and pack ql_nope, which the plan states it does not model.
_BF16_KV = PrecisionProfile.canonicalize({"cache": [
    {"select": {"slots": ["kv", "latent", "rope_key"]},
     "format": {"dtype": "BF16"}}]})


def _launches(model_id: str, framework: str, *, phase: str = "decode",
              precision: Any = PRESETS["w_fp8_kv_fp8"],
              ) -> Tuple[Dict[str, List[Tuple[str, ...]]], List[str]]:
    """Each of `_PLANS`' member sets, and the names of the casts left
    standalone, after the deployed binding's rewrite on one H100."""
    analysis = evaluate_step(
        resolve_snapshot(model_id), ParallelConfig(), phase=phase, batch=1,
        input_tokens=512, output_tokens=64, precision=precision).analysis
    binding = get_binding(framework).deployed(ServingDeployment.at(
        gpu_sku="H100-SXM", activation_dtype=analysis.precision.model_dtype))
    serving = rewrite_serving_graph(analysis, framework_binding=binding)
    pre = serving.pre_program
    launches: Dict[str, List[Tuple[str, ...]]] = {}
    for node in serving.program.nodes():
        if node.op in _PLANS:
            launches.setdefault(node.op, []).append(tuple(sorted(
                pre.by_uid(uid).name for uid in node.origins)))
    standalone = [node.name for node in serving.program.nodes()
                  if node.op == "quantise_activation"]
    return launches, standalone


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_plans_that_write_the_norm_are_tried_before_the_bare_norm(framework):
    """The loader applies plans in the pattern's own order (`match_order`),
    not file order: `fused_add_rmsnorm` and `fused_q_kv_rmsnorm` tie with
    `fused_rmsnorm_quant` on required nodes and win on all nodes, since each
    can take the norm's cast in. sglang first: it binds all three with no
    condition, so a swap there is what the default set sees."""
    order = [op for op, _impl in get_binding(framework).op_implementations]
    bare = order.index("fused_rmsnorm_quant")
    for larger in ("fused_add_rmsnorm", "fused_q_kv_rmsnorm"):
        assert order.index(larger) < bare, (larger, framework)


def test_module_bench_tries_plans_in_the_loader_order():
    """`module_bench` applies the union of the shipped plans, sorted with
    the loader's own key, so each binding's order is a subsequence of it."""
    from backend.playground.module_bench import shipped_fusion_plans
    union = [plan.scheme_name for plan in shipped_fusion_plans()]
    for framework in ("sglang", "vllm"):
        ops = list(dict.fromkeys(
            op for op, _impl in get_binding(framework).op_implementations))
        assert sorted(ops, key=union.index) == ops, framework


@pytest.mark.parametrize("framework", ["sglang", "vllm"])
def test_bare_norm_takes_its_cast_and_the_add_keeps_its_own(framework):
    """DeepSeek-V3 under FP8: `q_a_layernorm` and the cast `q_b_proj`
    reads are one launch, and the post-attention add + norm + cast stay
    `fused_add_rmsnorm`'s."""
    launches, standalone = _launches("deepseek-ai/DeepSeek-V3", framework)
    assert ("q_a_layernorm", "q_b_proj__cast") in launches.get(
        "fused_rmsnorm_quant", [])
    assert "q_b_proj__cast" not in standalone
    assert ("gate_proj__cast", "post_attention_layernorm",
            "post_attn_add") in launches["fused_add_rmsnorm"]
    assert all("post_attention_layernorm" not in members
               for members in launches["fused_rmsnorm_quant"])


def test_paired_norms_keep_the_pair_and_take_q_cast():
    """Where the q and kv a-norms are one launch (`fused_q_kv_rmsnorm`,
    Kimi-K3 under vLLM), that launch takes q's cast as its optional member;
    the bare-norm plan does not split the pair."""
    launches, _ = _launches("moonshotai/Kimi-K3", "vllm")
    pairs = launches.get("fused_q_kv_rmsnorm", [])
    assert ("kv_a_layernorm", "q_a_layernorm", "q_b_proj__cast") in pairs, pairs
    taken = {name for members in launches.get("fused_rmsnorm_quant", [])
             for name in members}
    assert not taken & {name for members in pairs for name in members}


@pytest.mark.parametrize("model_id", ["deepseek-ai/DeepSeek-V3.2",
                                      "zai-org/GLM-5"])
@pytest.mark.parametrize("phase", ["decode", "prefill"])
def test_vllm_dsa_query_is_one_launch(model_id, phase):
    """vLLM's `fused_q` rotates the main q_pe and the indexer query and
    quantises the latter in one launch, whatever the indexer's rope layout
    (V3.2 NeoX, GLM-5 interleaved): no standalone rope or cast remains on
    the query path. Deployed with a BF16 MLA cache (`_BF16_KV`), the path
    the plan prices."""
    launches, standalone = _launches(model_id, "vllm", phase=phase,
                                     precision=_BF16_KV)
    members = {"q_pe_rope", "indexer.q_pe_slice", "indexer.q_nope_slice",
               "indexer.q_pe_rope", "indexer.query_concat",
               "indexer_logits__cast"}
    fused = launches.get("fused_dsa_q_rope_quant", [])
    assert fused and all(set(m) == members for m in fused), fused
    assert "indexer_logits__cast" not in standalone
