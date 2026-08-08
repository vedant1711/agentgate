"""Efficiency and cost metrics — B5.

``est_cost_usd`` is 0.00 for every free tier and for local Ollama, and is computed anyway from
the price table. That is not decoration: it is how a run that cost nothing projects what the
same suite would cost at enterprise scale, which is the "minimal resources now, enterprise
later" argument stated as a number instead of a claim.

Latency is the sum of attributed step durations, not wall-clock — see
:class:`agentgate.schemas.trajectory.Trajectory`. A gate on wall-clock would fire on scheduling
noise and would be meaningless in replay mode, which is the mode CI uses.
"""

from __future__ import annotations

from typing import ClassVar

from agentgate.metrics.base import BaseMetric, Scored, ScoredSample
from agentgate.metrics.registry import register
from agentgate.schemas.common import Direction, DType, MetricFamily

EFFICIENCY = MetricFamily.EFFICIENCY
LOWER: Direction = "lower_is_better"


class _EfficiencyMetric(BaseMetric):
    """Shared declaration for the efficiency family."""

    family: ClassVar[MetricFamily] = EFFICIENCY
    direction: ClassVar[Direction] = LOWER


@register
class LatencyMs(_EfficiencyMetric):
    """Per-repetition latency in milliseconds."""

    name = "efficiency.latency_ms"
    dtype: ClassVar[DType] = "continuous"
    description: ClassVar[str] = (
        "Sum of attributed step durations. Reported as p50/p95 at suite level, because a mean "
        "latency hides exactly the tail that makes an agent unusable."
    )

    def compute(self, sample: ScoredSample) -> Scored:
        """Read the trajectory's attributed latency."""
        return Scored(
            value=sample.trajectory.latency_ms,
            detail={"steps": len(sample.trajectory.steps)},
        )


@register
class TotalTokens(_EfficiencyMetric):
    """Total tokens consumed by one repetition."""

    name = "efficiency.total_tokens"
    dtype: ClassVar[DType] = "count"
    description: ClassVar[str] = "Prompt plus completion tokens across every model round-trip."

    def compute(self, sample: ScoredSample) -> Scored:
        """Sum the trajectory's token usage."""
        return Scored(value=float(sample.trajectory.usage.total_tokens))


@register
class PromptTokens(_EfficiencyMetric):
    """Prompt tokens consumed by one repetition."""

    name = "efficiency.prompt_tokens"
    dtype: ClassVar[DType] = "count"
    description: ClassVar[str] = "Prompt tokens; rises when context or tool schemas grow."

    def compute(self, sample: ScoredSample) -> Scored:
        """Read prompt-token usage."""
        return Scored(value=float(sample.trajectory.usage.prompt_tokens))


@register
class CompletionTokens(_EfficiencyMetric):
    """Completion tokens produced by one repetition."""

    name = "efficiency.completion_tokens"
    dtype: ClassVar[DType] = "count"
    description: ClassVar[str] = (
        "Completion tokens. The metric a verbosity regression moves first, which is why the "
        "judge verbosity audit cross-checks against it."
    )

    def compute(self, sample: ScoredSample) -> Scored:
        """Read completion-token usage."""
        return Scored(value=float(sample.trajectory.usage.completion_tokens))


@register
class ToolCallsCount(_EfficiencyMetric):
    """Number of tool invocations in one repetition."""

    name = "efficiency.tool_calls_count"
    dtype: ClassVar[DType] = "count"
    description: ClassVar[str] = "Tool invocations, including failed ones — a retry still costs."

    def compute(self, sample: ScoredSample) -> Scored:
        """Count tool invocations."""
        return Scored(value=float(sample.trajectory.n_tool_calls))


@register
class LlmRoundtrips(_EfficiencyMetric):
    """Number of model round-trips in one repetition."""

    name = "efficiency.llm_roundtrips"
    dtype: ClassVar[DType] = "count"
    description: ClassVar[str] = "Model calls; the quantity a free tier's rate limit governs."

    def compute(self, sample: ScoredSample) -> Scored:
        """Count model round-trips."""
        return Scored(value=float(sample.trajectory.n_llm_roundtrips))


@register
class EstimatedCost(_EfficiencyMetric):
    """Projected USD cost of one repetition."""

    name = "efficiency.est_cost_usd"
    dtype: ClassVar[DType] = "continuous"
    description: ClassVar[str] = (
        "Cost from the price table. Zero on free tiers and Ollama by design; computed anyway so "
        "enterprise projection is a number rather than a claim (B5)."
    )

    def compute(self, sample: ScoredSample) -> Scored:
        """Read the projected cost recorded by the runner."""
        return Scored(value=sample.trajectory.est_cost_usd)
