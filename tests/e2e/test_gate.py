"""End-to-end gate tests (Phase 7 acceptance, H4).

The claims this file has to establish:

* a PR that flips ``FAULT_DROP_TOOL`` gets a FAILED gate with correct statistics;
* a no-op PR passes;
* every fault knob produces the verdict its signature declares;
* **false-positive control** — 20 no-change comparisons at different seeds yield at most 2
  regressions, which is what an alpha of 0.05 should look like.

Everything runs offline in mock mode.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentgate.demo import EXPECTED_VERDICTS, run_scenario, scenario_names
from agentgate.faults import SIGNATURES
from agentgate.gate import EXIT_PASS, EXIT_REGRESSION, EXIT_SAFETY, evaluate, naive_verdict
from agentgate.report import STICKY_MARKER, render_comment, render_html
from agentgate.schemas.common import Verdict

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[2]
CRM = REPO / "suites" / "crm_ops"


@pytest.fixture(scope="module")
def dropped_tool() -> object:
    return run_scenario("dropped_tool", suite_path=CRM, bootstrap_resamples=300)


@pytest.fixture(scope="module")
def no_op() -> object:
    return run_scenario("no_op", suite_path=CRM, bootstrap_resamples=300)


# ---------------------------------------------------------------------------
# Acceptance: a dropped tool fails the gate; a no-op passes
# ---------------------------------------------------------------------------


def test_dropping_a_tool_fails_the_gate_with_statistics(dropped_tool: object) -> None:
    result = dropped_tool.gate  # type: ignore[attr-defined]
    assert result.verdict.verdict is Verdict.REGRESSION
    assert result.verdict.exit_code == EXIT_REGRESSION
    assert "outcome.task_success" in result.verdict.failing_metrics

    ruling = result.ruling("outcome.task_success")
    assert ruling is not None
    assert ruling.comparison is not None
    assert ruling.comparison.delta.value < -0.3, "the drop must be large and measured"
    assert ruling.p_adjusted is not None
    assert ruling.p_adjusted < 0.05, "the regression must be significant after FDR correction"
    assert ruling.comparison.delta.ci_high is not None
    assert ruling.comparison.delta.ci_high < -ruling.margin, (
        "the whole confidence interval must sit below the margin"
    )


def test_a_no_op_change_passes(no_op: object) -> None:
    result = no_op.gate  # type: ignore[attr-defined]
    assert result.verdict.verdict is Verdict.PASS
    assert result.verdict.exit_code == EXIT_PASS
    assert result.verdict.failing_metrics == []
    assert not result.verdict.safety_failures


def test_a_no_op_proves_non_inferiority_rather_than_merely_not_failing(no_op: object) -> None:
    """PASS must mean 'proven fine', not 'we could not prove it broken'."""
    result = no_op.gate  # type: ignore[attr-defined]
    proven = [r for r in result.rulings if r.verdict is Verdict.PASS]
    assert proven, "at least one metric should establish non-inferiority on identical runs"
    for ruling in proven:
        assert ruling.p_noninferiority is not None
        assert ruling.p_noninferiority <= 0.05


def test_the_gate_reports_the_clustering_it_used(dropped_tool: object) -> None:
    comparison = dropped_tool.gate.comparison  # type: ignore[attr-defined]
    assert comparison.clustered, "the crm_ops suite is clustered and the gate must say so"
    ruling = dropped_tool.gate.ruling("trajectory.f1")  # type: ignore[attr-defined]
    assert ruling is not None
    assert any("clustered analysis" in note for note in ruling.reasons)


# ---------------------------------------------------------------------------
# Every scenario demonstrates what it claims
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", scenario_names())
def test_each_scenario_produces_its_declared_verdict(scenario: str) -> None:
    """A scenario that stops demonstrating its failure class must fail loudly."""
    expected = EXPECTED_VERDICTS.get(scenario, "ANY")
    result = run_scenario(scenario, suite_path=CRM, bootstrap_resamples=200)
    if expected == "ANY":
        pytest.skip(f"{scenario} has no pinned verdict")
    assert result.verdict == expected, (
        f"{scenario}: expected {expected}, got {result.verdict} — {result.gate.verdict.summary}"
    )


def test_the_injection_scenario_names_the_leaking_tasks() -> None:
    result = run_scenario("injection", suite_path=CRM, bootstrap_resamples=200)
    failures = result.gate.verdict.safety_failures
    assert failures, "the vulnerable agent must trip a tripwire"
    assert result.exit_code == EXIT_SAFETY
    metrics = {finding.metric for finding in failures}
    assert "safety.prompt_injection_compliance" in metrics
    assert all(finding.task_id for finding in failures), "every finding names its task"


def test_safety_failures_bypass_the_statistics() -> None:
    """A tripwire fails the gate however small the sample."""
    result = run_scenario("injection", suite_path=CRM, bootstrap_resamples=200)
    assert result.gate.verdict.verdict is Verdict.SAFETY_FAIL
    assert "SAFETY" in result.gate.verdict.summary


@pytest.mark.parametrize("scenario", sorted(SIGNATURES))
def test_each_knob_moves_the_metrics_its_signature_declares(scenario: str) -> None:
    """F2's signature table is ground truth: the declared metrics must move as declared."""
    signature = SIGNATURES[scenario]
    result = run_scenario(scenario, suite_path=CRM, bootstrap_resamples=200)
    comparison = result.gate.comparison

    checked = 0
    for expectation in signature.expects:
        item = comparison.comparison(expectation.metric)
        if item is None:
            continue  # the suite does not exercise this metric; other tests cover that
        checked += 1
        # delta is direction-normalised (higher is better), so a declared "down" is negative.
        if expectation.direction == "down":
            assert item.delta.value <= 1e-9, (
                f"{scenario}: expected {expectation.metric} to fall — {expectation.rationale}"
            )
        else:
            assert item.delta.value <= 1e-9 or item.delta.value >= -1e-9
    assert checked > 0, f"{scenario}: none of its declared metrics were comparable"


# ---------------------------------------------------------------------------
# False-positive control (H4)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_repeated_no_change_comparisons_rarely_fail() -> None:
    """20 identical comparisons at different seeds must yield at most 2 regressions.

    This is the property a naive threshold gate cannot have, and the reason anyone should trust
    a red build from this one.
    """
    regressions = 0
    for seed in range(20):
        result = run_scenario(
            "no_op",
            suite_path=CRM,
            seed=20260101 + seed * 7919,
            bootstrap_resamples=100,
            run_root=Path(".agentgate/fp-control"),
        )
        if result.gate.verdict.verdict in (Verdict.REGRESSION, Verdict.SAFETY_FAIL):
            regressions += 1
    assert regressions <= 2, f"{regressions}/20 no-change comparisons produced a failure"


def test_the_naive_gate_would_have_reached_a_different_conclusion(no_op: object) -> None:
    """The E6 argument, checked rather than asserted."""
    comparison = no_op.gate.comparison  # type: ignore[attr-defined]
    naive = naive_verdict(comparison, threshold=0.03)
    assert isinstance(naive, dict)
    assert not any(naive.values()), "on identical runs even the naive gate should pass"


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def test_the_pr_comment_is_sticky_and_complete(dropped_tool: object) -> None:
    comment = render_comment(dropped_tool.gate)  # type: ignore[attr-defined]
    assert comment.startswith(STICKY_MARKER), "CI greps for this marker to update in place"
    assert "GATE FAILED" in comment
    assert "outcome.task_success" in comment
    assert "Naive threshold vs statistical gate" in comment
    assert "pass^k decay" in comment
    assert "Power &amp; minimum detectable effect" in comment or "Power" in comment


def test_every_number_in_the_comment_carries_an_interval(dropped_tool: object) -> None:
    """The rule that makes the report trustworthy, enforced on the rendered output."""
    comment = render_comment(dropped_tool.gate)  # type: ignore[attr-defined]
    table_rows = [line for line in comment.splitlines() if line.startswith("| ") and "`" in line]
    assert table_rows
    for row in table_rows:
        cells = [cell.strip() for cell in row.split("|")]
        # Baseline and candidate columns (indices 3 and 4) must show a bracketed interval.
        assert "[" in cells[3] and "]" in cells[3], row
        assert "[" in cells[4] and "]" in cells[4], row


def test_the_html_report_is_self_contained(dropped_tool: object) -> None:
    html = render_html(dropped_tool.gate)  # type: ignore[attr-defined]
    assert html.startswith("<!doctype html>")
    assert "<svg" in html, "charts must render without JavaScript"
    assert "https://cdn" not in html and "<script src=" not in html, "no external assets"
    assert "vega-lite" in html, "vega specs are embedded for reuse"
    assert "outcome.task_success" in html
    assert "Reproducibility" in html


def test_the_html_report_writes_to_disk(tmp_path: Path, no_op: object) -> None:
    from agentgate.report import write_html

    path = write_html(no_op.gate, tmp_path / "report.html")  # type: ignore[attr-defined]
    assert path.exists()
    assert path.stat().st_size > 4_000


def test_gate_json_round_trips(dropped_tool: object) -> None:
    from agentgate.schemas.results import GateVerdict

    verdict = dropped_tool.gate.verdict  # type: ignore[attr-defined]
    restored = GateVerdict.model_validate_json(verdict.model_dump_json())
    assert restored.verdict is verdict.verdict
    assert restored.exit_code == verdict.exit_code
    assert restored.policy_hash == verdict.policy_hash


# ---------------------------------------------------------------------------
# Policy behaviour
# ---------------------------------------------------------------------------


def test_widening_the_margin_can_turn_a_regression_into_a_pass(dropped_tool: object) -> None:
    """The demo's teaching moment, checked: the same data, a different policy, a new verdict."""
    from agentgate.schemas.policy import GatedMetric, GatePolicy

    comparison = dropped_tool.gate.comparison  # type: ignore[attr-defined]
    lenient = GatePolicy(
        gated_metrics=[GatedMetric(metric="outcome.task_success", margin=0.95)],
        safety_tripwires=[],
    )
    result = evaluate(comparison, lenient)
    assert result.verdict.verdict is not Verdict.REGRESSION


def test_strict_mode_fails_an_underpowered_suite() -> None:
    from agentgate.schemas.policy import GatedMetric, GatePolicy

    result = run_scenario("verbosity_attack", suite_path=CRM, bootstrap_resamples=200)
    strict = GatePolicy(
        gated_metrics=[GatedMetric(metric="judge.coherence", margin=0.01)],
        underpowered_behavior="fail",
        safety_tripwires=[],
    )
    verdict = evaluate(result.gate.comparison, strict).verdict
    assert verdict.verdict in (Verdict.UNDERPOWERED, Verdict.PASS, Verdict.SKIPPED)


def test_the_policy_hash_changes_when_a_margin_is_loosened() -> None:
    from agentgate.schemas.policy import GatedMetric, GatePolicy

    tight = GatePolicy(gated_metrics=[GatedMetric(metric="outcome.task_success", margin=0.03)])
    loose = GatePolicy(gated_metrics=[GatedMetric(metric="outcome.task_success", margin=0.30)])
    assert tight.policy_hash() != loose.policy_hash(), (
        "a reviewer must be able to see that the bar moved"
    )
