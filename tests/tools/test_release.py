"""The release entry point keeps every producer on the deployed trace mode."""
from __future__ import annotations

import pytest

from scripts.release import _fly_trace_mode


def test_fly_is_the_single_trace_mode_source(tmp_path):
    fly = tmp_path / "fly.toml"
    fly.write_text(
        "[env]\nHA_CODE_TRACE = 'functions'\n",
        encoding="utf-8")
    assert _fly_trace_mode(fly) == "functions"


def test_an_unknown_trace_mode_is_refused_before_warm_or_publish(tmp_path):
    fly = tmp_path / "fly.toml"
    fly.write_text("[env]\nHA_CODE_TRACE = 'sidecar'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="HA_CODE_TRACE"):
        _fly_trace_mode(fly)
