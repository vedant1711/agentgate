"""Fault-config parsing and the signature table that acts as ground truth for H4."""

from __future__ import annotations

import pytest

from agentgate.faults import SIGNATURES, FaultConfig, scenario_names, signature
from agentgate.faults.config import ENV_PREFIX


def test_default_config_is_the_healthy_baseline() -> None:
    healthy = FaultConfig()
    assert not healthy.enabled
    assert healthy.active() == []
    assert healthy.truncate_context == 1.0


def test_env_parsing_covers_every_knob() -> None:
    env = {
        f"{ENV_PREFIX}PROMPT_DEGRADE": "1",
        f"{ENV_PREFIX}DROP_TOOL": "refund_order",
        f"{ENV_PREFIX}TRUNCATE_CONTEXT": "0.5",
        f"{ENV_PREFIX}MODEL_SWAP": "mock/agent-small",
        f"{ENV_PREFIX}TEMPERATURE": "1.2",
        f"{ENV_PREFIX}TOOL_LATENCY_MS": "250",
        f"{ENV_PREFIX}TOOL_ERROR_RATE": "0.15",
        f"{ENV_PREFIX}INJECTION_VULN": "true",
        f"{ENV_PREFIX}VERBOSITY": "yes",
    }
    config = FaultConfig.from_env(env)
    assert config.prompt_degrade
    assert config.drop_tool == "refund_order"
    assert config.truncate_context == 0.5
    assert config.model_swap == "mock/agent-small"
    assert config.temperature == 1.2
    assert config.tool_latency_ms == 250.0
    assert config.tool_error_rate == 0.15
    assert config.injection_vuln
    assert config.verbosity
    assert len(config.active()) == 9


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "FALSE"])
def test_falsey_env_values_do_not_enable_a_knob(value: str) -> None:
    assert not FaultConfig.from_env({f"{ENV_PREFIX}PROMPT_DEGRADE": value}).prompt_degrade


def test_malformed_values_fall_back_to_healthy_defaults() -> None:
    """A typo in CI must not silently look like a passing gate."""
    config = FaultConfig.from_env(
        {f"{ENV_PREFIX}TOOL_ERROR_RATE": "banana", f"{ENV_PREFIX}TRUNCATE_CONTEXT": "not-a-float"}
    )
    assert config.tool_error_rate == 0.0
    assert config.truncate_context == 1.0


def test_out_of_range_values_are_clamped() -> None:
    config = FaultConfig.from_env(
        {f"{ENV_PREFIX}TOOL_ERROR_RATE": "9", f"{ENV_PREFIX}TRUNCATE_CONTEXT": "5"}
    )
    assert config.tool_error_rate == 1.0
    assert config.truncate_context == 1.0


def test_empty_environment_yields_the_baseline() -> None:
    assert not FaultConfig.from_env({}).enabled


def test_active_knobs_are_sorted_for_a_stable_manifest() -> None:
    config = FaultConfig(verbosity=True, drop_tool="refund_order", prompt_degrade=True)
    assert config.active() == sorted(config.active())


def test_hides_tool_only_matches_the_named_tool() -> None:
    config = FaultConfig(drop_tool="refund_order")
    assert config.hides_tool("refund_order")
    assert not config.hides_tool("get_order")
    assert not FaultConfig().hides_tool("refund_order")


# ---------------------------------------------------------------------------
# Signature table
# ---------------------------------------------------------------------------


def test_every_scenario_declares_at_least_one_expectation() -> None:
    for name in scenario_names():
        sig = signature(name)
        assert sig.expects, f"{name} declares no expected metric movement"
        assert sig.simulates, f"{name} does not say what it simulates"
        assert sig.config.enabled, f"{name} carries a healthy config"


def test_every_scenario_has_a_gate_relevant_expectation() -> None:
    non_gated = {"verbosity_attack"}  # audited, not gated — the judge panel catches it
    for name in scenario_names():
        sig = signature(name)
        if name in non_gated:
            continue
        assert sig.gated_expectations, f"{name} moves nothing the gate watches"


def test_scenario_names_are_unique_and_match_their_keys() -> None:
    for key, sig in SIGNATURES.items():
        assert sig.scenario == key


def test_expectations_name_real_metric_families() -> None:
    families = {"outcome", "trajectory", "rag", "safety", "efficiency", "reliability", "judge"}
    for sig in SIGNATURES.values():
        for expectation in sig.expects:
            assert expectation.metric.split(".", 1)[0] in families, expectation.metric
            assert expectation.rationale, f"{expectation.metric} has no rationale"


def test_unknown_scenario_lists_the_known_ones() -> None:
    with pytest.raises(KeyError, match="known scenarios"):
        signature("nope")
