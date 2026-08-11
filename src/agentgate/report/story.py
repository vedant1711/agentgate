"""The demo's narrative layer: turning a gate verdict into something a reader can follow.

The first version of the interactive demo opened with a slider labelled "margin δ" and a chart of
per-cluster differences. Both are the right things to show *eventually*, and both are meaningless
to someone who has not yet been told what problem they solve. A reader who does not already
understand non-inferiority testing learned nothing, which made the demo decorative.

This module assembles the story that has to come first:

1. **A change someone actually made**, in one sentence of plain English.
2. **The raw before-and-after**, per task, as pass or fail — no statistics yet.
3. **What a normal CI check would conclude** from those totals, so the reader sees the naive
   answer before being told it is unreliable.
4. **The control**: the same agent run twice with nothing changed, which also moves. This is the
   hinge of the whole argument, and it needs to be shown rather than asserted.

Only after those does a margin slider mean anything.

Everything here is derived from a real scenario run through the real pipeline. The naive verdict
in particular is computed, not imagined — it is what a `if score_drop > threshold: fail` rule
would genuinely have done with these numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agentgate.schemas.results import MetricResult
from agentgate.stats.aggregate import repetition_scores, task_means

NAIVE_THRESHOLD = 0.03
"""The industry-standard rule this project exists to replace: fail if any metric drops >3 points.

Not a straw man. It is the rule that ships in most agent CI templates, and the reason it fails is
arithmetic rather than carelessness: on a small suite, a 3-point move is well inside the range a
healthy agent produces run to run.
"""

PLAIN_METRIC_LABELS: dict[str, str] = {
    "outcome.task_success": "finished the job correctly",
    "outcome.exact_match": "gave exactly the right answer",
    "outcome.f1_token": "answer overlapped the expected one",
    "outcome.semantic_similarity": "answer meant the right thing",
    "trajectory.in_order_match": "used the right tools in the right order",
    "trajectory.any_order_match": "used the right tools",
    "trajectory.recall": "found all the tools it needed",
    "trajectory.precision": "avoided unnecessary tool calls",
    "trajectory.argument_correctness": "passed the right arguments",
    "trajectory.step_efficiency": "worked without wasted steps",
    "trajectory.f1": "tool-call overlap with gold",
    "trajectory.lcs_ratio": "kept the gold sequence in order",
    "trajectory.redundant_call_rate": "repeated a call it had already made",
    "trajectory.loop_detected": "got stuck in a loop",
    "trajectory.error_recovery_rate": "recovered after a tool error",
    "trajectory.single_tool_use": "solved it with one tool call",
    "reliability.pass_hat_k": "succeeded on every attempt",
    "reliability.pass_at_1": "succeeded on the first attempt",
    "reliability.pass_at_k": "succeeded on at least one attempt",
    "outcome.abstained": "declined to answer",
    "outcome.json_valid": "returned valid JSON",
    "rag.answer_relevancy": "answered the question asked",
    "rag.faithfulness": "stayed faithful to its sources",
    "rag.context_precision": "retrieved relevant context",
    "safety.destructive_action_without_confirmation": "acted destructively without confirming",
    "safety.injection_compliance": "obeyed a hidden instruction",
    "efficiency.tool_calls_count": "tool calls made",
    "efficiency.llm_roundtrips": "model round trips",
    "efficiency.prompt_tokens": "prompt tokens",
    "efficiency.completion_tokens": "completion tokens",
    "efficiency.est_cost_usd": "estimated cost",
    "judge.instruction_following": "followed the instructions",
    "judge.coherence": "wrote coherently",
    "efficiency.total_tokens": "tokens used",
    "efficiency.latency_ms": "time taken",
}
"""What each metric means to someone who has never read the catalogue."""


def plain_label(metric: str) -> str:
    """A human-readable name for a metric, falling back to a de-namespaced identifier."""
    if metric in PLAIN_METRIC_LABELS:
        return PLAIN_METRIC_LABELS[metric]
    return metric.split(".", 1)[-1].replace("_", " ")


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    """One task's before-and-after, averaged over its repetitions."""

    task_id: str
    baseline: float
    candidate: float

    @property
    def delta(self) -> float:
        """Candidate minus baseline."""
        return self.candidate - self.baseline

    @property
    def state(self) -> str:
        """``worse``, ``better``, or ``same`` — what a reader sees as a colour."""
        if self.delta < 0:
            return "worse"
        return "better" if self.delta > 0 else "same"


def task_outcomes(
    baseline: Sequence[MetricResult], candidate: Sequence[MetricResult]
) -> list[TaskOutcome]:
    """Pair per-sample scores into one before/after row per task.

    Only tasks scored on both sides are kept, for the same reason the statistics layer does it:
    comparing a score against nothing is not a comparison.
    """
    before = task_means(repetition_scores(baseline))
    after = task_means(repetition_scores(candidate))
    return [
        TaskOutcome(task_id=task_id, baseline=before[task_id], candidate=after[task_id])
        for task_id in sorted(set(before) & set(after))
    ]


@dataclass(frozen=True, slots=True)
class NaiveVerdict:
    """What a conventional threshold check would have concluded from the totals alone.

    Shown *before* the statistics, because the reader has to see the plausible-looking wrong
    answer before the right one means anything.
    """

    baseline_rate: float
    candidate_rate: float
    threshold: float
    n_tasks: int

    @property
    def drop(self) -> float:
        """How far the headline number fell. Negative means it rose."""
        return self.baseline_rate - self.candidate_rate

    @property
    def would_block(self) -> bool:
        """True when a ``fail if it dropped more than the threshold`` rule fires.

        The epsilon is not pedantry. Rates are sums of floats, so a drop that is exactly the
        threshold lands at ``0.030000000000000027`` about as often as at ``0.03`` — and without
        the tolerance the naive rule's verdict would depend on floating-point dust rather than on
        the data. The rest of this codebase clamps for the same reason.
        """
        return self.drop > self.threshold + 1e-9

    @property
    def verdict(self) -> str:
        """The naive gate's answer."""
        return "BLOCK" if self.would_block else "SHIP"

    def describe(self) -> str:
        """One sentence stating what the naive rule saw and did."""
        direction = "fell" if self.drop > 0 else "rose"
        return (
            f"The score {direction} from {self.baseline_rate:.0%} to {self.candidate_rate:.0%} "
            f"across {self.n_tasks} tasks."
        )


def naive_verdict(
    outcomes: Sequence[TaskOutcome], *, threshold: float = NAIVE_THRESHOLD
) -> NaiveVerdict:
    """Apply the conventional rule to a set of task outcomes."""
    n = len(outcomes)
    if n == 0:
        return NaiveVerdict(0.0, 0.0, threshold, 0)
    return NaiveVerdict(
        baseline_rate=sum(item.baseline for item in outcomes) / n,
        candidate_rate=sum(item.candidate for item in outcomes) / n,
        threshold=threshold,
        n_tasks=n,
    )


# ---------------------------------------------------------------------------
# Scenario copy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScenarioCopy:
    """Plain-language framing for one demo scenario."""

    key: str
    title: str
    change: str
    """What a developer did, in one sentence."""
    why_it_matters: str
    lesson: str
    """What this scenario is here to teach."""


SCENARIO_COPY: dict[str, ScenarioCopy] = {
    "no_op": ScenarioCopy(
        key="no_op",
        title="You changed nothing at all",
        change="Nobody touched the agent. This is the exact same code, run twice.",
        why_it_matters=(
            "Agents are stochastic, so the score moves anyway. Every number you are about to "
            "see in the other scenarios has to be read against this one."
        ),
        lesson="A score that moved is not evidence that anything broke.",
    ),
    "dropped_tool": ScenarioCopy(
        key="dropped_tool",
        title="A refactor renamed a tool",
        change="Someone renamed a tool during a refactor and missed one registration.",
        why_it_matters=(
            "The agent can no longer do a thing it used to do. This is a genuine regression and "
            "a gate that misses it is worthless."
        ),
        lesson="When something is really broken, the evidence is unambiguous.",
    ),
    "verbosity_attack": ScenarioCopy(
        key="verbosity_attack",
        title="You asked the agent to be more thorough",
        change='Someone added "be thorough and explain your reasoning" to the system prompt.',
        why_it_matters=(
            "Answers got much longer. Longer answers score higher on some metrics for reasons "
            "that have nothing to do with being more correct."
        ),
        lesson="Some changes move the numbers without changing the quality.",
    ),
    "injection": ScenarioCopy(
        key="injection",
        title="A customer note contained a hidden instruction",
        change=(
            "A support ticket contained text telling the agent to ignore its rules — and the "
            "agent followed it."
        ),
        why_it_matters=(
            "This is a security failure, not a quality one. It does not matter how good the "
            "average score is."
        ),
        lesson="Some failures should never be averaged. They stop the build outright.",
    ),
    "model_downgrade": ScenarioCopy(
        key="model_downgrade",
        title="Finance asked you to use a cheaper model",
        change="The agent was switched to a smaller, cheaper model to cut the bill.",
        why_it_matters="The question is whether you gave up quality, and how much.",
        lesson="A cost saving is only a saving if you can show what it cost.",
    ),
    "prompt_degrade": ScenarioCopy(
        key="prompt_degrade",
        title="Someone simplified the system prompt",
        change='A "cleanup" PR deleted a paragraph of the system prompt that looked redundant.',
        why_it_matters="The deleted paragraph was the one enforcing an approval policy.",
        lesson="Prompt edits have side effects nobody predicts.",
    ),
    "flaky_dependency": ScenarioCopy(
        key="flaky_dependency",
        title="A downstream service got flaky",
        change="An internal API the agent calls started failing a fraction of the time.",
        why_it_matters="Nothing in your code changed, but your agent got worse.",
        lesson="Regressions do not always come from your own commits.",
    ),
    "context_truncation": ScenarioCopy(
        key="context_truncation",
        title="You shrank the context window to save money",
        change="The number of retrieved documents passed to the agent was reduced.",
        why_it_matters="Less context is cheaper, and sometimes it is also worse.",
        lesson="Cheaper is a trade, and a trade needs both numbers.",
    ),
    "sampling_drift": ScenarioCopy(
        key="sampling_drift",
        title="Someone raised the temperature",
        change="Sampling temperature was nudged up to make answers less repetitive.",
        why_it_matters="More variety means less consistency, and consistency is what users feel.",
        lesson="An average can hide a reliability problem completely.",
    ),
}


def copy_for(scenario: str) -> ScenarioCopy:
    """Plain-language framing for a scenario, with a safe generic fallback."""
    if scenario in SCENARIO_COPY:
        return SCENARIO_COPY[scenario]
    return ScenarioCopy(
        key=scenario,
        title=scenario.replace("_", " "),
        change="A change was made to the agent.",
        why_it_matters="The question is whether it made the agent worse.",
        lesson="Measure it rather than guessing.",
    )


VERDICT_PLAIN: dict[str, dict[str, str]] = {
    "PASS": {
        "headline": "Ship it",
        "body": (
            "The evidence shows this change did not make the agent meaningfully worse. Not "
            "merely 'we found no problem' — the suite was large enough to have found one."
        ),
    },
    "REGRESSION": {
        "headline": "Block the merge",
        "body": (
            "The agent really did get worse, by more than the amount you said you could "
            "tolerate. This is not noise."
        ),
    },
    "UNDERPOWERED": {
        "headline": "Not enough evidence",
        "body": (
            "The numbers moved, but not by enough to tell a real change from ordinary "
            "run-to-run variation. Add more test cases or accept a looser tolerance — but do "
            "not read this as a pass."
        ),
    },
    "SAFETY_FAIL": {
        "headline": "Stop — safety failure",
        "body": (
            "The agent did something it must never do. No amount of good average performance "
            "makes up for it, so the statistics are skipped entirely."
        ),
    },
    "INCONCLUSIVE": {
        "headline": "Nothing to rule on",
        "body": "No metric produced a comparison the gate could evaluate.",
    },
}


def verdict_copy(verdict: str) -> dict[str, str]:
    """Plain-language headline and explanation for a gate verdict."""
    return VERDICT_PLAIN.get(verdict, {"headline": verdict, "body": "See the report for details."})


COMPARABLE_DTYPES = frozenset({"binary", "proportion"})
"""Metrics whose deltas share a [-1, 1] scale and can honestly sit on one axis.

A forest plot puts every row on a **single** x-axis, so it is only meaningful when the rows are
commensurable. Token counts move in the hundreds and success rates in hundredths; drawing them on
one axis would either flatten every quality metric to an invisible sliver or need a second scale,
and a dual-scale chart is the fastest way to make a rigorous plot lie. Efficiency metrics are
therefore shown separately, in their own units.
"""


def forest_rows(rulings: Sequence[Any]) -> list[dict[str, Any]]:
    """Per-metric effect sizes with intervals, ready to plot as a forest plot.

    A forest plot is the right form here because the question is not "how big is each number" but
    "which of these intervals excludes the line" — the reader's eye performs the test.

    Args:
        rulings: The gate's per-metric rulings.

    Returns:
        One row per commensurable metric, worst effect first, each carrying its own verdict so
        colour is never the only channel.
    """
    rows: list[dict[str, Any]] = []
    for ruling in rulings:
        comparison = ruling.comparison
        if comparison.dtype not in COMPARABLE_DTYPES:
            continue
        delta = comparison.delta
        if delta.ci_low is None or delta.ci_high is None:
            continue
        rows.append(
            {
                "metric": comparison.metric,
                "label": plain_label(comparison.metric),
                "delta": round(delta.value, 4),
                "lo": round(delta.ci_low, 4),
                "hi": round(delta.ci_high, 4),
                "margin": round(ruling.margin, 4),
                "verdict": ruling.verdict,
                "blocks": bool(ruling.blocks),
                # Two different numbers, and conflating them is the anti-conservative error this
                # project exists to prevent. `n_pairs` counts paired *tasks*; the tests run on
                # per-cluster means, so the independent sample size is the number of clusters —
                # 14, not 70, on the crm_ops suite. Reporting the larger one overstates power
                # fivefold.
                "n_tasks": comparison.n_pairs,
                "n_units": len(comparison.analysis_units) or comparison.n_pairs,
                "clustered": bool(getattr(comparison, "clustered", False)),
                "p": None if ruling.p_adjusted is None else round(ruling.p_adjusted, 4),
                "baseline": round(comparison.baseline.value, 4),
                "candidate": round(comparison.candidate.value, 4),
            }
        )
    return sorted(rows, key=lambda row: row["delta"])


def efficiency_rows(rulings: Sequence[Any]) -> list[dict[str, Any]]:
    """Cost and latency metrics, kept off the forest plot because their units differ."""
    rows: list[dict[str, Any]] = []
    for ruling in rulings:
        comparison = ruling.comparison
        if comparison.dtype in COMPARABLE_DTYPES or not comparison.metric.startswith("efficiency"):
            continue
        rows.append(
            {
                "label": plain_label(comparison.metric),
                "baseline": round(comparison.baseline.value, 2),
                "candidate": round(comparison.candidate.value, 2),
                "delta": round(comparison.delta.value, 2),
                "verdict": ruling.verdict,
            }
        )
    return rows


def outcome_payload(outcomes: Sequence[TaskOutcome]) -> list[dict[str, Any]]:
    """Serialise task outcomes for the page."""
    return [
        {
            "id": item.task_id,
            "b": round(item.baseline, 4),
            "c": round(item.candidate, 4),
            "s": item.state,
        }
        for item in outcomes
    ]
