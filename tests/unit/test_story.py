"""Tests for the demo's narrative layer.

The published page makes claims in prose — "a normal CI check ships this security breach", "this
21-point drop isn't established". Those sentences are only true because of numbers computed
elsewhere, and prose does not fail a build when it stops matching reality.

So the load-bearing test here is :func:`test_the_pages_headline_claim_is_still_true`, which pins
the exact disagreements the landing page is built around.
"""

from __future__ import annotations

import pytest

from agentgate.report.story import (
    NAIVE_THRESHOLD,
    SCENARIO_COPY,
    TaskOutcome,
    copy_for,
    naive_verdict,
    outcome_payload,
    plain_label,
    task_outcomes,
    verdict_copy,
)
from agentgate.schemas import MetricFamily, MetricResult


def score(task_id: str, value: float, *, rep: int = 0, system: str = "baseline") -> MetricResult:
    return MetricResult(
        metric="outcome.task_success",
        task_id=task_id,
        rep=rep,
        system=system,
        value=value,
        family=MetricFamily.OUTCOME,
        dtype="binary",
    )


# ---------------------------------------------------------------------------
# Pairing before and after
# ---------------------------------------------------------------------------


def test_outcomes_pair_the_same_task_across_both_sides() -> None:
    outcomes = task_outcomes(
        [score("a", 1.0), score("b", 1.0)],
        [score("a", 0.0, system="candidate"), score("b", 1.0, system="candidate")],
    )
    assert [(o.task_id, o.baseline, o.candidate) for o in outcomes] == [
        ("a", 1.0, 0.0),
        ("b", 1.0, 1.0),
    ]
    assert [o.state for o in outcomes] == ["worse", "same"]


def test_a_task_only_one_side_ran_is_dropped_not_paired_against_nothing() -> None:
    outcomes = task_outcomes(
        [score("a", 1.0), score("only-baseline", 1.0)],
        [score("a", 1.0, system="candidate")],
    )
    assert [o.task_id for o in outcomes] == ["a"]


def test_repetitions_are_averaged_into_one_row_per_task() -> None:
    outcomes = task_outcomes(
        [score("a", 1.0, rep=0), score("a", 0.0, rep=1)],
        [score("a", 1.0, rep=0, system="c"), score("a", 1.0, rep=1, system="c")],
    )
    assert len(outcomes) == 1
    assert outcomes[0].baseline == pytest.approx(0.5)
    assert outcomes[0].candidate == pytest.approx(1.0)
    assert outcomes[0].state == "better"


# ---------------------------------------------------------------------------
# The naive rule, faithfully implemented
# ---------------------------------------------------------------------------


def test_the_naive_rule_blocks_on_a_drop_past_its_threshold() -> None:
    """Implemented honestly rather than as a straw man — it is the rule most teams ship."""
    outcomes = [TaskOutcome(f"t{i}", 1.0, 0.0 if i < 2 else 1.0) for i in range(10)]
    verdict = naive_verdict(outcomes)
    assert verdict.baseline_rate == pytest.approx(1.0)
    assert verdict.candidate_rate == pytest.approx(0.8)
    assert verdict.drop == pytest.approx(0.2)
    assert verdict.would_block
    assert verdict.verdict == "BLOCK"


def test_the_naive_rule_ships_when_the_average_did_not_move() -> None:
    outcomes = [TaskOutcome(f"t{i}", 1.0, 1.0) for i in range(10)]
    assert naive_verdict(outcomes).verdict == "SHIP"


def test_the_naive_rule_ignores_offsetting_movement_entirely() -> None:
    """The blind spot the demo exists to show: totals identical, every case changed."""
    outcomes = [
        TaskOutcome("a", 1.0, 0.0),
        TaskOutcome("b", 0.0, 1.0),
    ]
    verdict = naive_verdict(outcomes)
    assert verdict.drop == pytest.approx(0.0)
    assert verdict.verdict == "SHIP"
    assert all(o.state != "same" for o in outcomes)


def test_a_drop_exactly_on_the_threshold_does_not_block() -> None:
    outcomes = [TaskOutcome("a", 1.0, 1.0 - NAIVE_THRESHOLD)]
    assert not naive_verdict(outcomes).would_block


def test_an_empty_comparison_does_not_block() -> None:
    assert naive_verdict([]).verdict == "SHIP"


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------


def test_every_scenario_the_demo_shows_has_written_copy() -> None:
    """A missing entry would surface a raw key like ``verbosity_attack`` on the page."""
    from scripts.build_demo_site import SCENARIOS

    for key in SCENARIOS:
        assert key in SCENARIO_COPY, f"{key} has no plain-language copy"


def test_scenario_copy_never_leaks_an_identifier() -> None:
    for key, copy in SCENARIO_COPY.items():
        assert key not in copy.title
        assert copy.title[0].isupper()
        assert copy.change.endswith((".", "!", "?"))


def test_an_unknown_scenario_falls_back_without_crashing() -> None:
    fallback = copy_for("something_new")
    assert fallback.title == "something new"
    assert fallback.change


def test_every_verdict_has_plain_language() -> None:
    for verdict in ("PASS", "REGRESSION", "UNDERPOWERED", "SAFETY_FAIL", "INCONCLUSIVE"):
        copy = verdict_copy(verdict)
        assert copy["headline"]
        assert copy["body"]
        assert verdict not in copy["headline"], "the headline should not restate the enum"


def test_metric_labels_are_readable_and_fall_back_gracefully() -> None:
    assert plain_label("outcome.task_success") == "finished the job correctly"
    assert plain_label("some.brand_new_metric") == "brand new metric"


def test_outcome_payload_is_compact_and_json_ready() -> None:
    payload = outcome_payload([TaskOutcome("a", 1.0, 0.0)])
    assert payload == [{"id": "a", "b": 1.0, "c": 0.0, "s": "worse"}]


# ---------------------------------------------------------------------------
# The claim the landing page is built on
# ---------------------------------------------------------------------------


def test_the_pages_headline_claim_is_still_true() -> None:
    """The demo asserts in prose that the naive check is wrong in *both* directions.

    That sentence is only true because of numbers, and prose does not fail a build when it stops
    matching reality. If a change to the suite, the agent, or the policy ever made these two
    scenarios agree with a threshold rule, the published page would be making a false claim about
    its own output — so this test fails first.

    Deliberately not marked slow: it runs in a couple of seconds, and CI is exactly where a
    published claim silently going stale needs to be caught.
    """
    from pathlib import Path

    from agentgate.demo import run_scenario

    suite = Path("suites/crm_ops")
    stop = {"REGRESSION", "SAFETY_FAIL"}

    # A security breach that a threshold check would happily ship.
    injection = run_scenario("injection", suite_path=suite)
    outcomes = task_outcomes(
        [s for s in injection.baseline_scores if s.metric == "outcome.task_success"],
        [s for s in injection.candidate_scores if s.metric == "outcome.task_success"],
    )
    assert not naive_verdict(outcomes).would_block, "a threshold check must ship this"
    assert injection.verdict in stop, "AgentGate must stop it"

    # A visible drop that a threshold check would block but the evidence does not establish.
    verbosity = run_scenario("verbosity_attack", suite_path=suite)
    outcomes = task_outcomes(
        [s for s in verbosity.baseline_scores if s.metric == "outcome.task_success"],
        [s for s in verbosity.candidate_scores if s.metric == "outcome.task_success"],
    )
    assert naive_verdict(outcomes).would_block, "a threshold check must block this"
    assert verbosity.verdict not in stop, "AgentGate must not call it a regression"
