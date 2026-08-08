"""Trajectory metrics — B2.

Each dimension is its own metric and is never blended into a single "trajectory score". NVIDIA's
guidance is explicit about why: a blended number lets a collapse in tool selection hide behind
good argument accuracy, and the whole point of trajectory evaluation is that the final answer
already hides things.

Several metrics here deliberately **skip** rather than score a default. ``error_recovery_rate``
on a run with no errors is undefined, not perfect; scoring it 1.0 would reward flawless runs
twice and make the suite mean move whenever the error *rate* changed.
"""

from __future__ import annotations

from typing import ClassVar

from agentgate.metrics.base import (
    BaseMetric,
    Scored,
    ScoredSample,
    binary,
    ratio,
)
from agentgate.metrics.matching import TrajectoryMatch, lcs_length, match_trajectory
from agentgate.metrics.registry import register, register_instance
from agentgate.schemas.common import Direction, DType, MetricFamily, Requirement
from agentgate.schemas.trajectory import ToolInvocation

TRAJECTORY = MetricFamily.TRAJECTORY
LOOP_THRESHOLD = 3
"""Consecutive identical (tool, args) calls that count as a loop (B2)."""
RECOVERY_WINDOW = 2
"""How many subsequent calls may contain the corrective action (B2 heuristic)."""


def _match(sample: ScoredSample) -> TrajectoryMatch:
    """Align the trajectory against its reference, using the sample's embedder for arguments."""
    reference = sample.reference.trajectory
    if reference is None:  # pragma: no cover - guarded by `requires`
        msg = "task declares no reference trajectory"
        raise ValueError(msg)
    return match_trajectory(
        reference, sample.trajectory.tool_invocations, embedder=sample.context.embedder
    )


class _MatchMetric(BaseMetric):
    """Shared plumbing for metrics derived from the reference alignment."""

    family: ClassVar[MetricFamily] = TRAJECTORY
    requires: ClassVar[set[Requirement]] = {Requirement.REFERENCE_TRAJECTORY}

    def detail_of(self, match: TrajectoryMatch, sample: ScoredSample) -> dict[str, object]:
        """Audit trail shared by the alignment metrics."""
        reference = sample.reference.trajectory
        steps = reference.steps if reference is not None else []
        predicted = sample.trajectory.tool_sequence
        return {
            "predicted": predicted,
            "reference": [step.tools for step in steps],
            "missing": [steps[i].tools[0] for i in match.missing_refs],
            "extra": [predicted[i] for i in match.extra_preds],
        }


@register
class ExactMatch(_MatchMetric):
    """Predicted tool-call sequence identical to the reference: same calls, same order."""

    name = "trajectory.exact_match"
    dtype: ClassVar[DType] = "binary"
    description: ClassVar[str] = "Identical call sequence, no extras, no omissions."

    def compute(self, sample: ScoredSample) -> Scored:
        """Check the strictest alignment."""
        match = _match(sample)
        return Scored(value=binary(match.exact_match), detail=self.detail_of(match, sample))


@register
class InOrderMatch(_MatchMetric):
    """Reference sequence appears as a subsequence of the prediction; extras allowed."""

    name = "trajectory.in_order_match"
    dtype: ClassVar[DType] = "binary"
    description: ClassVar[str] = (
        "Every required reference call appears, in order. Unordered groups may be satisfied in "
        "any relative order; optional steps never count against the match."
    )

    def compute(self, sample: ScoredSample) -> Scored:
        """Check the ordered alignment."""
        match = _match(sample)
        return Scored(value=binary(match.in_order_match), detail=self.detail_of(match, sample))


@register
class AnyOrderMatch(_MatchMetric):
    """All reference calls present, in any order; extras allowed."""

    name = "trajectory.any_order_match"
    dtype: ClassVar[DType] = "binary"
    description: ClassVar[str] = "Every required reference call appears somewhere."

    def compute(self, sample: ScoredSample) -> Scored:
        """Check the unordered alignment."""
        match = _match(sample)
        return Scored(value=binary(match.any_order_match), detail=self.detail_of(match, sample))


@register
class Precision(_MatchMetric):
    """Correct predicted calls / total predicted calls."""

    name = "trajectory.precision"
    dtype: ClassVar[DType] = "proportion"
    description: ClassVar[str] = "Share of the agent's calls that the reference asked for."

    def compute(self, sample: ScoredSample) -> Scored:
        """Score predicted-call precision."""
        match = _match(sample)
        return Scored(
            value=match.precision,
            detail={"matched": len(match.matched_preds), "predicted": match.n_predicted},
        )


@register
class Recall(_MatchMetric):
    """Reference calls recovered / total required reference calls."""

    name = "trajectory.recall"
    dtype: ClassVar[DType] = "proportion"
    description: ClassVar[str] = "Share of required reference calls the agent actually made."

    def compute(self, sample: ScoredSample) -> Scored:
        """Score reference-call recall."""
        match = _match(sample)
        return Scored(
            value=match.recall,
            detail={"matched": len(match.matched_refs), "required": match.n_reference},
        )


@register
class TrajectoryF1(_MatchMetric):
    """Harmonic mean of trajectory precision and recall."""

    name = "trajectory.f1"
    dtype: ClassVar[DType] = "proportion"
    description: ClassVar[str] = "Harmonic mean of trajectory precision and recall."

    def compute(self, sample: ScoredSample) -> Scored:
        """Score trajectory F1."""
        match = _match(sample)
        return Scored(value=match.f1, detail={"precision": match.precision, "recall": match.recall})


@register
class LcsRatio(_MatchMetric):
    """LCS(predicted, reference) / len(reference) — order-sensitive partial credit."""

    name = "trajectory.lcs_ratio"
    dtype: ClassVar[DType] = "proportion"
    description: ClassVar[str] = (
        "Longest common subsequence over tool names, honouring allowed alternatives, divided by "
        "the reference length. The one metric that rewards getting the order partly right."
    )

    def compute(self, sample: ScoredSample) -> Scored:
        """Score the LCS ratio."""
        reference = sample.reference.trajectory
        assert reference is not None
        required = [step for step in reference.steps if not step.optional]
        if not required:
            return Scored.skip("reference trajectory declares no required steps")
        length = lcs_length(required, sample.trajectory.tool_invocations)
        return Scored(value=ratio(length, len(required)), detail={"lcs": length})


@register
class ArgumentCorrectness(_MatchMetric):
    """Correct key-value parameter pairs / expected pairs, matched by tool name."""

    name = "trajectory.argument_correctness"
    dtype: ClassVar[DType] = "proportion"
    description: ClassVar[str] = (
        "Per-field comparison (exact, numeric tolerance, regex, semantic) over the arguments the "
        "reference declares. Arguments never supplied count as wrong."
    )

    def compute(self, sample: ScoredSample) -> Scored:
        """Score argument correctness."""
        match = _match(sample)
        value = match.argument_correctness
        if value is None:
            return Scored.skip("reference trajectory declares no expected arguments")
        return Scored(
            value=value,
            detail={
                "checks": [
                    {
                        "key": check.key,
                        "expected": check.expected,
                        "actual": check.actual,
                        "ok": check.ok,
                        "comparator": check.comparator,
                    }
                    for check in match.arg_checks
                ]
            },
        )


class SingleToolUse(BaseMetric):
    """Was a required tool invoked at least once (B2, spec-parameterised)?

    Args:
        tool: A specific tool to check. When omitted, the metric checks every tool the task's
            reference trajectory lists in ``required_tools``.
    """

    family: ClassVar[MetricFamily] = TRAJECTORY
    dtype: ClassVar[DType] = "binary"
    requires: ClassVar[set[Requirement]] = {Requirement.REFERENCE_TRAJECTORY}
    description: ClassVar[str] = "Required tool(s) invoked at least once."

    def __init__(self, tool: str | None = None) -> None:
        self.tool = tool
        self.name = (
            "trajectory.single_tool_use" if tool is None else f"trajectory.single_tool_use[{tool}]"
        )

    def compute(self, sample: ScoredSample) -> Scored:
        """Check that every required tool was used."""
        reference = sample.reference.trajectory
        assert reference is not None
        wanted = [self.tool] if self.tool is not None else list(reference.required_tools)
        if not wanted:
            return Scored.skip("reference trajectory declares no required_tools")
        used = set(sample.trajectory.tool_sequence)
        missing = [tool for tool in wanted if tool not in used]
        return Scored(value=binary(not missing), detail={"required": wanted, "missing": missing})


register_instance(SingleToolUse())


@register
class StepEfficiency(BaseMetric):
    """len(reference) / len(predicted), capped at 1 — penalises redundant or looping steps."""

    name = "trajectory.step_efficiency"
    family: ClassVar[MetricFamily] = TRAJECTORY
    dtype: ClassVar[DType] = "continuous"
    requires: ClassVar[set[Requirement]] = {Requirement.REFERENCE_TRAJECTORY}
    description: ClassVar[str] = (
        "Reference step count over predicted step count, capped at 1 so being *shorter* than "
        "the reference is not rewarded — a short trajectory that skipped work is a recall "
        "failure, and recall already measures it."
    )

    def compute(self, sample: ScoredSample) -> Scored:
        """Score step efficiency."""
        reference = sample.reference.trajectory
        assert reference is not None
        expected = reference.required_step_count
        actual = len(sample.trajectory.tool_invocations)
        if expected == 0:
            return Scored.skip("reference trajectory declares no required steps")
        if actual == 0:
            return Scored(value=0.0, detail={"predicted_steps": 0, "reference_steps": expected})
        return Scored(
            value=min(1.0, expected / actual),
            detail={"predicted_steps": actual, "reference_steps": expected},
        )


@register
class RedundantCallRate(BaseMetric):
    """Duplicate (tool, args) invocations / total invocations."""

    name = "trajectory.redundant_call_rate"
    family: ClassVar[MetricFamily] = TRAJECTORY
    dtype: ClassVar[DType] = "proportion"
    direction: ClassVar[Direction] = "lower_is_better"
    description: ClassVar[str] = (
        "Share of calls that repeat an identical (tool, arguments) pair already made."
    )

    def compute(self, sample: ScoredSample) -> Scored:
        """Count repeated call signatures."""
        calls = sample.trajectory.tool_invocations
        if not calls:
            return Scored.skip("no tool calls to assess")
        signatures = [call.signature for call in calls]
        duplicates = len(signatures) - len(set(signatures))
        return Scored(
            value=ratio(duplicates, len(signatures)),
            detail={"duplicates": duplicates, "total": len(signatures)},
        )


@register
class LoopDetected(BaseMetric):
    """Three or more consecutive repetitions of an identical (tool, args) call."""

    name = "trajectory.loop_detected"
    family: ClassVar[MetricFamily] = TRAJECTORY
    dtype: ClassVar[DType] = "binary"
    direction: ClassVar[Direction] = "lower_is_better"
    description: ClassVar[str] = (
        f"{LOOP_THRESHOLD}+ consecutive identical (tool, arguments) calls — the agent is stuck."
    )

    def compute(self, sample: ScoredSample) -> Scored:
        """Scan for a run of identical consecutive calls."""
        calls = sample.trajectory.tool_invocations
        longest, current = _longest_run(calls)
        return Scored(
            value=binary(longest >= LOOP_THRESHOLD),
            detail={"longest_run": longest, "signature": current},
        )


@register
class ErrorRecoveryRate(BaseMetric):
    """Of tool calls that errored, the fraction followed by a corrective action that succeeds."""

    name = "trajectory.error_recovery_rate"
    family: ClassVar[MetricFamily] = TRAJECTORY
    dtype: ClassVar[DType] = "proportion"
    description: ClassVar[str] = (
        f"Heuristic: a failed call counts as recovered when the same tool succeeds within the "
        f"next {RECOVERY_WINDOW} calls. Skipped (not scored 1.0) when nothing failed, because "
        f"'never had to recover' is not the same as 'recovered well'."
    )

    def compute(self, sample: ScoredSample) -> Scored:
        """Score recovery from failed tool calls."""
        calls = sample.trajectory.tool_invocations
        failures = [index for index, call in enumerate(calls) if not call.ok]
        if not failures:
            return Scored.skip("no failed tool calls in this trajectory")
        recovered = sum(1 for index in failures if _recovered(calls, index))
        return Scored(
            value=ratio(recovered, len(failures)),
            detail={"failures": len(failures), "recovered": recovered},
        )


@register
class GroundedReasoning(BaseMetric):
    """TRACE-style: fraction of factual claims in reasoning attributable to prior tool output."""

    name = "trajectory.grounded_reasoning"
    family: ClassVar[MetricFamily] = TRAJECTORY
    dtype: ClassVar[DType] = "proportion"
    requires: ClassVar[set[Requirement]] = {Requirement.JUDGE}
    description: ClassVar[str] = (
        "Judge-assisted claim extraction over intermediate reasoning, checked against an "
        "evidence bank of prior tool outputs and retrieved contexts (TRACE, arXiv:2510.02837)."
    )

    def compute(self, sample: ScoredSample) -> Scored:
        """Extract reasoning claims and check each against the evidence bank."""
        judge = sample.context.judge
        if judge is None:  # pragma: no cover - guarded by `requires`
            return Scored.skip("no judge configured")
        reasoning = " ".join(sample.trajectory.reasoning_texts)
        if not reasoning.strip():
            return Scored.skip("trajectory records no intermediate reasoning")
        claims = judge.extract_claims(reasoning)
        if not claims:
            return Scored.skip("no factual claims found in the reasoning")
        evidence = sample.trajectory.evidence_bank
        supported = judge.check_claims(claims, evidence)
        return Scored(
            value=ratio(sum(supported), len(claims)),
            detail={
                "claims": claims,
                "supported": [bool(flag) for flag in supported],
                "evidence_items": len(evidence),
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _longest_run(calls: list[ToolInvocation]) -> tuple[int, str]:
    """Return the longest run of identical consecutive call signatures, and its signature."""
    longest = 0
    longest_signature = ""
    current = 0
    previous = ""
    for call in calls:
        signature = call.signature
        current = current + 1 if signature == previous else 1
        previous = signature
        if current > longest:
            longest, longest_signature = current, signature
    return longest, longest_signature


def _recovered(calls: list[ToolInvocation], index: int) -> bool:
    """Did the agent fix the failure at ``index`` within the recovery window?"""
    failed = calls[index]
    window = calls[index + 1 : index + 1 + RECOVERY_WINDOW]
    return any(call.tool == failed.tool and call.ok for call in window)
