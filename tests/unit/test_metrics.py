"""Metrics engine tests (Phase 4 acceptance).

The claims: every registered metric is covered by hand-computed golden fixtures, the engine
skips rather than zero-scores unscoreable samples, and the generated catalogue matches the code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentgate.errors import ConfigError
from agentgate.metrics import (
    HashingEmbedder,
    LexicalJudge,
    MetricContext,
    MetricsEngine,
    ScoringConfig,
    cosine,
    group_by_metric,
    registry,
    score_run,
    scored_metrics,
    subset_match,
)
from agentgate.metrics.checkers import extract_number, is_abstention, normalize_answer
from agentgate.metrics.docgen import metrics_doc_is_current, render_metrics_doc
from agentgate.metrics.embeddings import get_embedder
from agentgate.metrics.outcome import validate_against_schema
from agentgate.metrics.trajectory import SingleToolUse
from agentgate.runner import load_suite
from agentgate.schemas.common import MetricFamily, Requirement
from tests.metric_fixtures import GoldenCase, load_all_cases, sample_for

REPO = Path(__file__).resolve().parents[2]
GOLDEN_CASES = load_all_cases()
CONTEXT = MetricContext(judge=LexicalJudge(), embedder=HashingEmbedder())


# ---------------------------------------------------------------------------
# Golden values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda case: case.label)
def test_golden_value(case: GoldenCase) -> None:
    """Every committed fixture reproduces its hand-computed expectation."""
    result = registry.get(case.metric).score(sample_for(case, CONTEXT))
    assert result.status == case.status, f"{case.label}: {result.error or result.detail}"
    if case.expected is None:
        assert result.value is None
    else:
        assert result.value == pytest.approx(case.expected), case.note


def test_every_registered_metric_has_golden_coverage() -> None:
    """Phase 4 acceptance: 100% of metrics covered by golden-value tests."""
    covered = {case.metric for case in GOLDEN_CASES}
    missing = sorted(set(registry.names()) - covered)
    assert missing == [], f"metrics without golden fixtures: {missing}"


def test_every_metric_has_at_least_three_cases() -> None:
    """H1 requires perfect / partial / degenerate coverage per metric."""
    counts: dict[str, int] = {}
    for case in GOLDEN_CASES:
        counts[case.metric] = counts.get(case.metric, 0) + 1
    thin = sorted(name for name, count in counts.items() if count < 3)
    assert thin == [], f"metrics with fewer than 3 golden cases: {thin}"


def test_golden_cases_include_degenerate_inputs() -> None:
    """At least one case per fixture file must exercise a skip or an empty input."""
    skipped = [case for case in GOLDEN_CASES if case.expected is None]
    assert len(skipped) >= 15, "requirement-skip behaviour is under-tested"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_is_populated_and_consistent() -> None:
    assert len(registry.names()) >= 40
    for metric in registry.all_metrics():
        assert metric.name
        assert isinstance(metric.family, MetricFamily)
        assert metric.dtype in ("binary", "proportion", "continuous", "count")
        assert metric.direction in ("higher_is_better", "lower_is_better")
        assert all(isinstance(item, Requirement) for item in metric.requires)


def test_every_family_except_reliability_has_metrics() -> None:
    for family in MetricFamily:
        metrics = registry.by_family(family)
        if family is MetricFamily.RELIABILITY:
            assert metrics == [], "reliability is computed by the statistics engine, not here"
        else:
            assert metrics, f"no metrics registered for {family.value}"


def test_unknown_metric_lists_the_known_ones() -> None:
    with pytest.raises(ConfigError, match="registered metrics"):
        registry.get("outcome.not_a_metric")


def test_duplicate_registration_is_refused() -> None:
    with pytest.raises(ConfigError, match="duplicate metric name"):
        registry.register_instance(SingleToolUse())


def test_safety_metrics_are_all_binary_and_lower_is_better() -> None:
    """A tripwire that is not binary cannot be a tripwire."""
    for metric in registry.by_family(MetricFamily.SAFETY):
        assert metric.dtype == "binary", metric.name
        assert metric.direction == "lower_is_better", metric.name


def test_parameterised_single_tool_use_names_its_tool() -> None:
    metric = SingleToolUse("refund_order")
    assert metric.name == "trajectory.single_tool_use[refund_order]"


# ---------------------------------------------------------------------------
# Engine behaviour
# ---------------------------------------------------------------------------


def test_engine_scores_a_real_run_end_to_end() -> None:
    import asyncio

    from agentgate.agents import build_agent

    suite = load_suite(REPO / "suites" / "smoke")
    agent = build_agent("tool_agent")
    trajectories = [asyncio.run(agent.run(task, 7)) for task in suite.tasks]

    results = score_run(suite, trajectories, run_id="r1")
    grouped = group_by_metric(results)

    assert len(grouped) == len(registry.names())
    assert all(len(bucket) == len(suite.tasks) for bucket in grouped.values())
    assert "outcome.task_success" in scored_metrics(results)

    successes = [item.value for item in grouped["outcome.task_success"] if item.is_scored]
    assert successes == [1.0] * len(suite.tasks), "the baseline agent passes the smoke suite"


def test_engine_skips_rather_than_zero_scores() -> None:
    """The rule that keeps a task-authoring gap from looking like a regression."""
    import asyncio

    from agentgate.agents import build_agent

    suite = load_suite(REPO / "suites" / "smoke")
    agent = build_agent("tool_agent")
    trajectories = [asyncio.run(agent.run(task, 7)) for task in suite.tasks]
    results = score_run(suite, trajectories, run_id="r1")

    retrieval_metrics = {
        "rag.faithfulness",
        "rag.hallucination_rate",
        "rag.context_precision",
        "rag.context_recall",
    }
    rag_results = [item for item in results if item.metric in retrieval_metrics]
    assert rag_results, "RAG metrics must still be attempted"
    assert all(item.status == "skipped" for item in rag_results), (
        "a CRM suite retrieves nothing; the retrieval metrics must skip, not score 0"
    )
    assert all(item.value is None for item in rag_results)


def test_engine_can_run_a_subset_of_metrics() -> None:
    engine = MetricsEngine(ScoringConfig(metrics=["outcome.task_success", "trajectory.f1"]))
    assert engine.metric_names == ["outcome.task_success", "trajectory.f1"]


def test_disabling_the_judge_skips_judge_backed_metrics() -> None:
    case = next(case for case in GOLDEN_CASES if case.metric == "judge.coherence")
    context = MetricContext(judge=None, embedder=HashingEmbedder())
    result = registry.get("judge.coherence").score(sample_for(case, context))
    assert result.status == "skipped"
    assert "judge" in str(result.detail)


def test_budget_halted_units_score_nothing() -> None:
    from agentgate.schemas.trajectory import RunStatus

    case = next(case for case in GOLDEN_CASES if case.metric == "efficiency.total_tokens")
    halted = case.trajectory.model_copy(update={"status": RunStatus.BUDGET_EXHAUSTED})
    sample = sample_for(case, CONTEXT).__class__(
        task=case.task, trajectory=halted, run_id="r", context=CONTEXT
    )
    result = registry.get("efficiency.total_tokens").score(sample)
    assert result.status == "skipped"
    assert "not executed" in str(result.detail)


def test_a_raising_metric_is_recorded_as_an_error_not_a_crash() -> None:
    from agentgate.metrics.base import BaseMetric, Scored, ScoredSample

    class Exploding(BaseMetric):
        name = "test.exploding"
        family = MetricFamily.OUTCOME
        dtype = "binary"

        def compute(self, sample: ScoredSample) -> Scored:  # noqa: ARG002
            msg = "kaboom"
            raise RuntimeError(msg)

    case = GOLDEN_CASES[0]
    result = Exploding().score(sample_for(case, CONTEXT))
    assert result.status == "error"
    assert "kaboom" in (result.error or "")
    assert result.value is None


def test_out_of_range_values_are_clamped_and_flagged() -> None:
    from agentgate.metrics.base import BaseMetric, Scored, ScoredSample

    class Overflowing(BaseMetric):
        name = "test.overflowing"
        family = MetricFamily.OUTCOME
        dtype = "proportion"

        def compute(self, sample: ScoredSample) -> Scored:  # noqa: ARG002
            return Scored(value=1.7)

    result = Overflowing().score(sample_for(GOLDEN_CASES[0], CONTEXT))
    assert result.value == 1.0
    assert result.detail["clamped_from"] == 1.7


# ---------------------------------------------------------------------------
# Supporting machinery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The Refund Was Processed!", "refund was processed"),
        ("  a  b  ", "b"),
        ("", ""),
    ],
)
def test_answer_normalisation(text: str, expected: str) -> None:
    assert normalize_answer(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [("42 units", 42.0), ("$1,234.50 total", 1234.5), ("-7 degrees", -7.0), ("none", None)],
)
def test_number_extraction(text: str, expected: float | None) -> None:
    assert extract_number(text) == expected


def test_abstention_detection() -> None:
    assert is_abstention("I don't know based on the documentation.")
    assert not is_abstention("The answer is 42.")


@pytest.mark.parametrize(
    ("expected", "actual", "ok"),
    [
        ({"a": 1}, {"a": 1, "b": 2}, True),
        ({"a": 1}, {"b": 2}, False),
        ([{"to": "x"}], [{"to": "y"}, {"to": "x"}], True),
        ([{"to": "x"}], [{"to": "y"}], False),
        (45.0, 45.001, True),
        (45.0, 46.0, False),
    ],
)
def test_goal_state_subset_matching(expected: object, actual: object, ok: bool) -> None:
    assert subset_match(expected, actual, tolerance=0.01) is ok


def test_schema_validation_reports_every_problem() -> None:
    schema = {
        "type": "object",
        "required": ["a", "b"],
        "properties": {"a": {"type": "integer", "minimum": 0}},
    }
    problems = validate_against_schema({"a": -1}, schema)
    assert any("missing required property 'b'" in problem for problem in problems)
    assert any("minimum" in problem for problem in problems)


def test_unknown_schema_keywords_are_ignored_not_failed() -> None:
    schema = {"type": "object", "someFutureKeyword": {"x": 1}}
    assert validate_against_schema({"a": 1}, schema) == []


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder()
    first = embedder.encode(["the refund was processed"])[0]
    second = embedder.encode(["the refund was processed"])[0]
    assert first == second
    assert cosine(first, second) == pytest.approx(1.0)
    assert cosine(first, embedder.encode(["completely different words here"])[0]) < 0.3


def test_embedder_factory_caches() -> None:
    assert get_embedder("hashing-bow") is get_embedder("hashing-bow")


def test_bigrams_give_some_word_order_sensitivity() -> None:
    embedder = HashingEmbedder()
    forward, backward = embedder.encode(["alice refunded bob", "bob refunded alice"])
    assert cosine(forward, backward) < 1.0


def test_lexical_judge_is_deterministic() -> None:
    judge = LexicalJudge()
    claims = judge.extract_claims("The target is 95 percent. The owner is the platform lead.")
    assert len(claims) == 2
    assert judge.check_claims(claims, ["The target is 95 percent."]) == [True, False]


# ---------------------------------------------------------------------------
# Generated documentation
# ---------------------------------------------------------------------------


def test_metric_catalogue_is_generated_from_the_registry() -> None:
    rendered = render_metrics_doc()
    for metric in registry.all_metrics():
        assert f"`{metric.name}`" in rendered
    assert f"{len(registry.names())} metrics registered" in rendered


def test_committed_metric_catalogue_is_current() -> None:
    assert metrics_doc_is_current(REPO / "docs" / "metrics.md"), (
        "run `agentgate docs metrics` and commit the result"
    )
