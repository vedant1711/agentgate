"""Rubric-judge metrics — B1's judge family.

Each criterion is scored **separately**. There is no blended "quality" number anywhere in
AgentGate, because decomposed rubric criteria beat open-ended scoring (G-Eval, arXiv:2303.16634)
and because a single number cannot tell you *which* thing got worse — which is the only
actionable part of a regression report.

Every result carries the judge's per-item samples and variance, so the statistics engine can
fold judge uncertainty into the metric's standard error via the law of total variance (C5).
"""

from __future__ import annotations

from typing import ClassVar

from agentgate.metrics.base import BaseMetric, JudgeVerdict, Scored, ScoredSample
from agentgate.metrics.registry import register
from agentgate.schemas.common import DType, MetricFamily, Requirement

OUTCOME = MetricFamily.OUTCOME


class _JudgeMetric(BaseMetric):
    """Shared plumbing: call one criterion, record its samples and variance."""

    family: ClassVar[MetricFamily] = OUTCOME
    dtype: ClassVar[DType] = "continuous"
    requires: ClassVar[set[Requirement]] = {Requirement.JUDGE}
    criterion: ClassVar[str] = ""

    def compute(self, sample: ScoredSample) -> Scored:
        """Score this metric's criterion."""
        judge = sample.context.judge
        if judge is None:  # pragma: no cover - guarded by `requires`
            return Scored.skip("no judge configured")
        verdict: JudgeVerdict = judge.score_criterion(
            self.criterion,
            sample.task.prompt,
            sample.answer,
            reference=sample.reference.answer or "",
            contexts=list(sample.trajectory.retrieved_contexts),
        )
        return Scored(
            value=verdict.value,
            detail={
                "judge": judge.name,
                "criterion": self.criterion,
                "rationale": verdict.rationale,
                "n_judge_samples": len(verdict.samples),
            },
            cost_tokens=verdict.cost_tokens,
            judge_samples=verdict.samples,
            judge_variance=verdict.variance,
        )


@register
class JudgeCorrectness(_JudgeMetric):
    """Is the answer factually right, judged against the reference?"""

    name = "judge.correctness"
    criterion: ClassVar[str] = "correctness"
    requires: ClassVar[set[Requirement]] = {Requirement.JUDGE, Requirement.REFERENCE_ANSWER}
    description: ClassVar[str] = "Rubric criterion: factual correctness vs the reference answer."


@register
class JudgeCompleteness(_JudgeMetric):
    """Does the answer cover everything the reference covers?"""

    name = "judge.completeness"
    criterion: ClassVar[str] = "completeness"
    requires: ClassVar[set[Requirement]] = {Requirement.JUDGE, Requirement.REFERENCE_ANSWER}
    description: ClassVar[str] = "Rubric criterion: coverage of the reference answer's content."


@register
class JudgeInstructionFollowing(_JudgeMetric):
    """Did the answer do what the prompt asked?"""

    name = "judge.instruction_following"
    criterion: ClassVar[str] = "instruction_following"
    description: ClassVar[str] = "Rubric criterion: adherence to the instruction as given."


@register
class JudgeCoherence(_JudgeMetric):
    """Is the answer internally consistent and non-repetitive?"""

    name = "judge.coherence"
    criterion: ClassVar[str] = "coherence"
    description: ClassVar[str] = (
        "Rubric criterion: internal consistency and absence of repetition. Note that a verbosity "
        "attack lowers this while an unmitigated judge would raise overall preference — which is "
        "exactly why criteria are scored separately (E4)."
    )
