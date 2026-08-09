"""Interactive-demo tests.

One property matters more than the rest: **the published page must never disagree with the tool
it demonstrates.** A demo that showed PASS where the gate said REGRESSION would discredit the
whole project, and it is the failure mode a hand-written JavaScript reimplementation of the
statistics would eventually produce. So the grid is evaluated by the real engine, and these tests
pin that it agrees with the gate at the gate's own settings.
"""

from __future__ import annotations

import json

import pytest

from agentgate.demo import ScenarioResult, run_scenario
from agentgate.report.playground import (
    ALPHA_CHOICES,
    MARGIN_STEPS,
    MetricPanel,
    build_panel,
    margin_steps,
    panel_payload,
    panels_for,
)
from agentgate.schemas.common import Verdict


@pytest.fixture(scope="module")
def dropped_tool() -> ScenarioResult:
    """A scenario the gate rules REGRESSION on, run through the real pipeline."""
    return run_scenario("dropped_tool")


@pytest.fixture(scope="module")
def panels(dropped_tool: ScenarioResult) -> list[MetricPanel]:
    return panels_for(dropped_tool.gate, limit=4)


def test_the_grid_agrees_with_the_gate_at_the_gates_own_settings(
    dropped_tool: ScenarioResult, panels: list[MetricPanel]
) -> None:
    """The demo and the gate must rule identically where they overlap.

    This is the guarantee that justifies precomputing with the engine rather than reimplementing
    the tests in the browser. Where the grid's margin and alpha match the policy's, the verdict
    must match the gate's ruling for that metric.
    """
    gate = dropped_tool.gate
    alpha = gate.policy.alpha
    assert alpha in ALPHA_CHOICES, "the demo must offer the policy's own alpha"

    checked = 0
    for panel in panels:
        ruling = gate.ruling(panel.metric)
        if ruling is None:
            continue
        matching = [
            point
            for point in panel.grid
            if point.alpha == alpha and point.margin == panel.default_margin
        ]
        assert matching, (
            f"{panel.metric}: the policy's own margin {panel.default_margin} is not a reachable "
            f"slider position, so the page cannot show what the gate actually decided"
        )
        for point in matching:
            assert point.verdict == ruling.verdict.value, (
                f"{panel.metric}: demo says {point.verdict}, gate says {ruling.verdict.value}"
            )
            checked += 1
    assert checked > 0, "the agreement check compared nothing"


def test_every_panel_evaluates_the_full_grid(panels: list[MetricPanel]) -> None:
    for panel in panels:
        assert len(panel.grid) == MARGIN_STEPS * len(ALPHA_CHOICES)
        assert len({point.margin for point in panel.grid}) == MARGIN_STEPS


def test_the_margin_slider_starts_at_zero_and_increases(panels: list[MetricPanel]) -> None:
    """A zero position must exist: 'prove exact equality' is the lesson the demo is built on."""
    for panel in panels:
        margins = sorted({point.margin for point in panel.grid})
        assert margins[0] == pytest.approx(0.0)
        assert margins[-1] > margins[0]


def test_panels_are_chosen_for_changing_the_answer_not_for_effect_size(
    panels: list[MetricPanel],
) -> None:
    """Ranking by |delta| surfaces token counts, whose verdicts never move."""
    diversity = [len({point.verdict for point in panel.grid}) for panel in panels]
    assert diversity == sorted(diversity, reverse=True)
    assert diversity[0] >= 2, "the leading panel must actually change verdict somewhere"


def test_a_metric_without_analysis_units_is_omitted_rather_than_shown_inert(
    dropped_tool: ScenarioResult,
) -> None:
    """A slider that cannot move implies stability that was never computed."""
    gate = dropped_tool.gate
    built = {panel.metric for panel in panels_for(gate, limit=99)}
    for comparison in gate.comparison.metrics:
        if len(comparison.analysis_units) < 2:
            assert comparison.metric not in built


def test_the_three_way_rule_is_the_gates_rule(dropped_tool: ScenarioResult) -> None:
    """Every grid verdict is one of the three the gate can reach from statistics alone.

    SAFETY_FAIL is deliberately absent: tripwires bypass the statistics, so a margin slider must
    never be able to produce one.
    """
    gate = dropped_tool.gate
    allowed = {Verdict.REGRESSION.value, Verdict.PASS.value, Verdict.UNDERPOWERED.value}
    for panel in panels_for(gate, limit=99):
        assert {point.verdict for point in panel.grid} <= allowed


def test_the_payload_round_trips_as_json_and_carries_the_real_units(
    panels: list[MetricPanel],
) -> None:
    payload = panel_payload(panels, scenario="dropped_tool", suite="crm_ops")
    restored = json.loads(json.dumps(payload))
    assert restored["scenario"] == "dropped_tool"
    assert restored["alphas"] == list(ALPHA_CHOICES)
    first = restored["panels"][0]
    assert len(first["units"]) == panels[0].n_units
    assert first["units"] == [round(value, 6) for value in panels[0].units]


def test_the_payload_contains_nothing_that_could_break_out_of_a_script_tag(
    panels: list[MetricPanel],
) -> None:
    """The page embeds this JSON inline, so a stray closing tag would be an injection."""
    blob = json.dumps(panel_payload(panels, scenario="dropped_tool", suite="crm_ops"))
    for dangerous in ("</script", "<script", "<!--", "]]>"):
        assert dangerous not in blob


def test_a_wider_margin_never_makes_a_verdict_more_severe(panels: list[MetricPanel]) -> None:
    """Monotonicity: loosening the margin cannot turn PASS into REGRESSION.

    A slider that flip-flopped would mean the underlying tests disagreed with themselves. The
    severity order is REGRESSION > UNDERPOWERED > PASS as the margin widens.
    """
    severity = {Verdict.REGRESSION.value: 2, Verdict.UNDERPOWERED.value: 1, Verdict.PASS.value: 0}
    for panel in panels:
        for alpha in ALPHA_CHOICES:
            series = [point for point in panel.grid if point.alpha == alpha]
            series.sort(key=lambda point: point.margin)
            levels = [severity[point.verdict] for point in series]
            assert levels == sorted(levels, reverse=True), (
                f"{panel.metric} at alpha={alpha} is not monotone in the margin: {levels}"
            )


def test_a_stricter_alpha_never_makes_a_regression_easier_to_declare(
    panels: list[MetricPanel],
) -> None:
    """Lowering alpha demands more evidence, so REGRESSION can only become rarer."""
    for panel in panels:
        by_margin: dict[float, dict[float, str]] = {}
        for point in panel.grid:
            by_margin.setdefault(point.margin, {})[point.alpha] = point.verdict
        for verdicts in by_margin.values():
            strict, loose = verdicts[min(ALPHA_CHOICES)], verdicts[max(ALPHA_CHOICES)]
            if strict == Verdict.REGRESSION.value:
                assert loose == Verdict.REGRESSION.value


def test_build_panel_respects_an_explicit_margin_ceiling(dropped_tool: ScenarioResult) -> None:
    gate = dropped_tool.gate
    comparison = next(item for item in gate.comparison.metrics if len(item.analysis_units) >= 2)
    panel = build_panel(comparison, policy=gate.policy, max_margin=0.5)
    assert max(point.margin for point in panel.grid) == pytest.approx(0.5)


def test_the_policy_margin_becomes_a_slider_position_without_losing_the_endpoints() -> None:
    """Snapping must not eat zero or the top of the range."""
    steps = margin_steps(1.0, include=0.37)
    assert len(steps) == MARGIN_STEPS
    assert steps == sorted(steps)
    assert steps[0] == pytest.approx(0.0)
    assert steps[-1] == pytest.approx(1.0)
    assert 0.37 in steps


def test_a_margin_outside_the_range_is_ignored_rather_than_distorting_the_scale() -> None:
    plain = margin_steps(1.0)
    assert margin_steps(1.0, include=5.0) == plain
    assert margin_steps(1.0, include=-1.0) == plain
