"""Plan order is most specific first; the plan's name only makes it
deterministic. Two equally specific plans that both match one node set
would be decided by their names, so the rewrite refuses them instead
(`eval.fusion_order`)."""
from __future__ import annotations

import pytest

from backend.arch.eval.fusion import Framework, Fusion
from backend.arch.eval.transforms import Deployment
from backend.inference.fusion_plans import FUSION_SCHEMES
from tests.snapshots import shared_program, shared_walk


def _walk():
    return shared_walk(Deployment(shared_program(
        "meta-llama__Llama-3.2-1B")).at(batch=1, input_tokens=16))


def test_two_equally_specific_plans_on_one_node_set_are_refused():
    """The same pattern bound under two names: equal specificity, the same
    matches — which one fuses would be alphabetical."""
    analysis = _walk()
    scheme = FUSION_SCHEMES["fused_add_rmsnorm"]
    framework = Framework("twins", (Fusion("a_twin", scheme),
                                    Fusion("b_twin", scheme)))
    with pytest.raises(ValueError, match="'a_twin' and 'b_twin'"):
        framework.rewrite(analysis.program, analysis)


def test_one_plan_alone_is_not_refused():
    analysis = _walk()
    framework = Framework("one", (Fusion.named(
        "fused_add_rmsnorm", FUSION_SCHEMES),))
    _program, rewrite = framework.rewrite(analysis.program, analysis)
    assert rewrite.origins
