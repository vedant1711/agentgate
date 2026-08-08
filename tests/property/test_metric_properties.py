"""Property-based metric invariants (H2).

Golden fixtures pin specific values; these check the laws that must hold for *every* input.
Between them they cover the two ways a metric goes wrong: a miscomputed number, and a number
that is right on the examples someone thought of.
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from agentgate.metrics import MetricContext, registry
from agentgate.metrics.base import ScoredSample
from agentgate.metrics.embeddings import HashingEmbedder
from agentgate.metrics.lexical_judge import LexicalJudge
from agentgate.metrics.matching import lcs_length, match_trajectory
from agentgate.schemas.task import ReferenceSpec, ReferenceStep, ReferenceTrajectory, TaskSpec
from agentgate.schemas.trajectory import (
    FinalStep,
    ToolCallStep,
    ToolInvocation,
    ToolResultStep,
    Trajectory,
)

TOOLS = ["lookup_customer", "get_order", "refund_order", "search_kb", "send_email", "calculator"]
CONTEXT = MetricContext(judge=LexicalJudge(), embedder=HashingEmbedder())

tool_names = st.sampled_from(TOOLS)
tool_lists = st.lists(tool_names, min_size=0, max_size=8)
nonempty_tool_lists = st.lists(tool_names, min_size=1, max_size=8)


def build_reference(tools: list[str]) -> ReferenceTrajectory:
    """A gold trajectory of single-alternative steps."""
    return ReferenceTrajectory(steps=[ReferenceStep(tools=[tool]) for tool in tools])


def build_invocations(tools: list[str]) -> list[ToolInvocation]:
    """Predicted invocations with distinct call ids."""
    return [
        ToolInvocation(index=i, call_id=f"c{i}", tool=tool, args={"i": i})
        for i, tool in enumerate(tools)
    ]


def build_sample(reference_tools: list[str], predicted_tools: list[str]) -> ScoredSample:
    """A scored sample carrying both trajectories."""
    task = TaskSpec(
        id="p1",
        cluster_id="c",
        inputs={"prompt": "do the thing"},
        reference=ReferenceSpec(answer="done", trajectory=build_reference(reference_tools)),
    )
    trajectory = Trajectory(task_id="p1", rep=0, seed=1, system="baseline", final_answer="done")
    for index, tool in enumerate(predicted_tools):
        call_id = f"c{index}"
        trajectory.add_step(ToolCallStep(index=0, call_id=call_id, tool=tool, args={}))
        trajectory.add_step(ToolResultStep(index=0, call_id=call_id, tool=tool, ok=True))
    trajectory.add_step(FinalStep(index=0, answer="done"))
    return ScoredSample(task=task, trajectory=trajectory, context=CONTEXT)


def score(metric_name: str, sample: ScoredSample) -> float | None:
    """Score one metric on a sample."""
    return registry.get(metric_name).score(sample).value


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(reference=tool_lists, predicted=tool_lists)
def test_all_proportion_and_binary_metrics_stay_in_range(
    reference: list[str], predicted: list[str]
) -> None:
    """No proportion or binary metric may ever leave [0, 1]."""
    sample = build_sample(reference, predicted)
    for metric in registry.all_metrics():
        if metric.dtype not in ("binary", "proportion"):
            continue
        value = metric.score(sample).value
        if value is not None:
            assert 0.0 <= value <= 1.0, f"{metric.name} produced {value}"


@settings(max_examples=200, deadline=None)
@given(reference=tool_lists, predicted=tool_lists)
def test_binary_metrics_only_ever_produce_zero_or_one(
    reference: list[str], predicted: list[str]
) -> None:
    sample = build_sample(reference, predicted)
    for metric in registry.all_metrics():
        if metric.dtype != "binary":
            continue
        value = metric.score(sample).value
        assert value in (None, 0.0, 1.0), f"{metric.name} produced {value}"


@settings(max_examples=200, deadline=None)
@given(reference=tool_lists, predicted=tool_lists)
def test_count_metrics_are_non_negative(reference: list[str], predicted: list[str]) -> None:
    sample = build_sample(reference, predicted)
    for metric in registry.all_metrics():
        if metric.dtype != "count":
            continue
        value = metric.score(sample).value
        assert value is None or value >= 0.0, f"{metric.name} produced {value}"


# ---------------------------------------------------------------------------
# The implication chain (H2)
# ---------------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(reference=nonempty_tool_lists, predicted=tool_lists)
def test_exact_implies_in_order_implies_any_order(
    reference: list[str], predicted: list[str]
) -> None:
    """exact_match => in_order_match => any_order_match, for every trajectory pair."""
    sample = build_sample(reference, predicted)
    exact = score("trajectory.exact_match", sample)
    in_order = score("trajectory.in_order_match", sample)
    any_order = score("trajectory.any_order_match", sample)
    if exact == 1.0:
        assert in_order == 1.0, (reference, predicted)
    if in_order == 1.0:
        assert any_order == 1.0, (reference, predicted)


@settings(max_examples=300, deadline=None)
@given(reference=nonempty_tool_lists, predicted=tool_lists)
def test_exact_match_implies_perfect_precision_and_recall(
    reference: list[str], predicted: list[str]
) -> None:
    sample = build_sample(reference, predicted)
    if score("trajectory.exact_match", sample) != 1.0:
        return
    assert score("trajectory.precision", sample) == 1.0
    assert score("trajectory.recall", sample) == 1.0
    assert score("trajectory.f1", sample) == 1.0


# ---------------------------------------------------------------------------
# Invariances
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(
    reference=nonempty_tool_lists,
    predicted=nonempty_tool_lists,
    permutation=st.randoms(use_true_random=False),
)
def test_any_order_match_is_invariant_under_permutation(
    reference: list[str], predicted: list[str], permutation: object
) -> None:
    """Permuting the predicted calls cannot change an order-insensitive metric."""
    import random as _random

    shuffled = list(predicted)
    _random.Random(12345).shuffle(shuffled)
    straight = build_sample(reference, predicted)
    scrambled = build_sample(reference, shuffled)
    assert score("trajectory.any_order_match", straight) == score(
        "trajectory.any_order_match", scrambled
    )
    assert score("trajectory.precision", straight) == score("trajectory.precision", scrambled)
    assert score("trajectory.recall", straight) == score("trajectory.recall", scrambled)


@settings(max_examples=200, deadline=None)
@given(reference=nonempty_tool_lists, predicted=tool_lists)
def test_precision_recall_f1_are_mutually_consistent(
    reference: list[str], predicted: list[str]
) -> None:
    sample = build_sample(reference, predicted)
    precision = score("trajectory.precision", sample)
    recall = score("trajectory.recall", sample)
    f1 = score("trajectory.f1", sample)
    assert precision is not None and recall is not None and f1 is not None
    if precision + recall == 0:
        assert f1 == 0.0
    else:
        assert f1 == pytest_approx(2 * precision * recall / (precision + recall))
    assert min(precision, recall) <= f1 <= max(precision, recall) + 1e-12


@settings(max_examples=200, deadline=None)
@given(reference=nonempty_tool_lists)
def test_lcs_ratio_is_monotone_under_appending_the_next_reference_call(
    reference: list[str],
) -> None:
    """Adding the next correct call can only help the LCS ratio."""
    steps = build_reference(reference).steps
    for prefix_len in range(len(reference)):
        shorter = build_invocations(reference[:prefix_len])
        longer = build_invocations(reference[: prefix_len + 1])
        assert lcs_length(steps, longer) >= lcs_length(steps, shorter)


@settings(max_examples=200, deadline=None)
@given(reference=nonempty_tool_lists, predicted=tool_lists)
def test_lcs_never_exceeds_either_sequence(reference: list[str], predicted: list[str]) -> None:
    steps = build_reference(reference).steps
    length = lcs_length(steps, build_invocations(predicted))
    assert 0 <= length <= min(len(reference), len(predicted))


@settings(max_examples=200, deadline=None)
@given(tools=nonempty_tool_lists)
def test_identical_trajectories_score_perfectly(tools: list[str]) -> None:
    """A prediction identical to its reference must be perfect on every alignment metric."""
    sample = build_sample(tools, tools)
    assert score("trajectory.exact_match", sample) == 1.0
    assert score("trajectory.in_order_match", sample) == 1.0
    assert score("trajectory.any_order_match", sample) == 1.0
    assert score("trajectory.f1", sample) == 1.0
    assert score("trajectory.lcs_ratio", sample) == 1.0
    assert score("trajectory.step_efficiency", sample) == 1.0


@settings(max_examples=200, deadline=None)
@given(reference=nonempty_tool_lists, extra=nonempty_tool_lists)
def test_appending_noise_never_improves_precision(reference: list[str], extra: list[str]) -> None:
    clean = build_sample(reference, reference)
    noisy = build_sample(reference, reference + extra)
    clean_precision = score("trajectory.precision", clean)
    noisy_precision = score("trajectory.precision", noisy)
    assert clean_precision is not None and noisy_precision is not None
    assert noisy_precision <= clean_precision + 1e-12


@settings(max_examples=200, deadline=None)
@given(reference=nonempty_tool_lists, extra=nonempty_tool_lists)
def test_appending_calls_never_improves_step_efficiency(
    reference: list[str], extra: list[str]
) -> None:
    clean = score("trajectory.step_efficiency", build_sample(reference, reference))
    noisy = score("trajectory.step_efficiency", build_sample(reference, reference + extra))
    assert clean is not None and noisy is not None
    assert noisy <= clean + 1e-12


# ---------------------------------------------------------------------------
# Matcher-level properties
# ---------------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(reference=tool_lists, predicted=tool_lists)
def test_matcher_accounting_is_internally_consistent(
    reference: list[str], predicted: list[str]
) -> None:
    match = match_trajectory(build_reference(reference), build_invocations(predicted))
    assert len(match.matched_preds) + len(match.extra_preds) == len(predicted)
    assert len(match.matched_refs) + len(match.missing_refs) == len(reference)
    assert match.ordered_matched_refs <= match.matched_refs | set(match.missing_refs)


@settings(max_examples=200, deadline=None)
@given(text=st.text(min_size=0, max_size=200))
def test_embedding_cosine_stays_in_range(text: str) -> None:
    from agentgate.metrics.matching import cosine

    embedder = HashingEmbedder()
    vectors = embedder.encode([text, text])
    assert 0.0 <= cosine(vectors[0], vectors[1]) <= 1.0


@settings(max_examples=100, deadline=None)
@given(text=st.text(min_size=1, max_size=200))
def test_identical_text_has_cosine_one(text: str) -> None:
    from agentgate.metrics.embeddings import _TOKEN_RE
    from agentgate.metrics.matching import cosine

    # Text with no tokens has no representation at all; the deliberate behaviour there is 0.0,
    # so that an empty answer never scores as similar to an empty reference.
    assume(_TOKEN_RE.findall(text.lower()))
    embedder = HashingEmbedder()
    vectors = embedder.encode([text, text])
    assert cosine(vectors[0], vectors[1]) == pytest_approx(1.0)


def pytest_approx(value: float) -> object:
    """Local alias so the helper reads naturally inside property assertions."""
    import pytest

    return pytest.approx(value, abs=1e-9)
