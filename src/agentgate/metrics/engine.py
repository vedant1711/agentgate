"""The metrics engine: score a run's trajectories with every applicable metric.

The engine's only real responsibilities are wiring services into :class:`MetricContext` and
never letting one metric's failure take down a run. It deliberately does *not* aggregate —
means, intervals, and reliability curves belong to the statistics engine (Part C), which needs
per-sample values to do its job.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from agentgate.metrics import registry
from agentgate.metrics.base import Embedder, Judge, Metric, MetricContext, ScoredSample
from agentgate.metrics.embeddings import get_embedder
from agentgate.metrics.lexical_judge import LexicalJudge
from agentgate.schemas.results import MetricResult
from agentgate.schemas.task import SuiteSpec, TaskSpec
from agentgate.schemas.trajectory import Trajectory

# Importing the metric modules is what populates the registry.
from agentgate.metrics import (  # noqa: F401  isort:skip
    efficiency,
    judge_metrics,
    outcome,
    rag,
    safety,
    trajectory as trajectory_metrics,
)


@dataclass(slots=True)
class ScoringConfig:
    """How a run is scored.

    Args:
        metrics: Metric names to run; ``None`` means every registered metric.
        judge: Judge backing the judge-dependent metrics. Defaults to the deterministic
            :class:`~agentgate.metrics.lexical_judge.LexicalJudge` so a suite scores offline for
            free; pass ``None`` explicitly to skip judge-backed metrics entirely.
        embedder: Embedder for similarity metrics. Defaults to the deterministic hashing encoder.
        options: Free-form per-metric options exposed through :class:`MetricContext`.
    """

    metrics: list[str] | None = None
    judge: Judge | None = field(default_factory=LexicalJudge)
    embedder: Embedder | None = field(default_factory=get_embedder)
    options: dict[str, Any] = field(default_factory=dict)


class MetricsEngine:
    """Scores trajectories against a suite.

    Args:
        config: Scoring configuration. Defaults score everything with offline services.
    """

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self.config = config or ScoringConfig()
        self.context = MetricContext(
            judge=self.config.judge,
            embedder=self.config.embedder,
            options=self.config.options,
        )
        self.metrics: list[Metric] = registry.select(self.config.metrics)

    def score_sample(
        self, task: TaskSpec, trajectory: Trajectory, *, run_id: str = ""
    ) -> list[MetricResult]:
        """Score one (task, repetition) with every selected metric.

        Args:
            task: The task specification.
            trajectory: The recorded execution.
            run_id: Owning run, recorded on each result.

        Returns:
            One :class:`MetricResult` per metric, including skipped and errored ones — an
            omitted result and a skipped result mean different things to the report.
        """
        sample = ScoredSample(task=task, trajectory=trajectory, run_id=run_id, context=self.context)
        return [metric.score(sample) for metric in self.metrics]

    def score_run(
        self, suite: SuiteSpec, trajectories: Iterable[Trajectory], *, run_id: str = ""
    ) -> list[MetricResult]:
        """Score every trajectory in a run.

        Args:
            suite: The suite the trajectories came from.
            trajectories: Recorded executions.
            run_id: Owning run.

        Returns:
            Flat list of results across all samples and metrics.

        Raises:
            KeyError: When a trajectory references a task the suite does not declare, which
                means the trajectories and the suite are out of sync.
        """
        results: list[MetricResult] = []
        for record in trajectories:
            task = suite.task(record.task_id)
            results.extend(self.score_sample(task, record, run_id=run_id))
        return results

    @property
    def metric_names(self) -> list[str]:
        """Names of the metrics this engine runs, in registry order."""
        return [metric.name for metric in self.metrics]


def score_run(
    suite: SuiteSpec,
    trajectories: Iterable[Trajectory],
    *,
    run_id: str = "",
    config: ScoringConfig | None = None,
) -> list[MetricResult]:
    """Convenience wrapper around :meth:`MetricsEngine.score_run`.

    Args:
        suite: The suite the trajectories came from.
        trajectories: Recorded executions.
        run_id: Owning run.
        config: Scoring configuration.

    Returns:
        Flat list of metric results.
    """
    return MetricsEngine(config).score_run(suite, trajectories, run_id=run_id)


def group_by_metric(results: Iterable[MetricResult]) -> dict[str, list[MetricResult]]:
    """Bucket results by metric name, preserving sample order within each bucket."""
    grouped: dict[str, list[MetricResult]] = {}
    for result in results:
        grouped.setdefault(result.metric, []).append(result)
    return grouped


def scored_metrics(results: Iterable[MetricResult]) -> list[str]:
    """Metric names that produced at least one usable value."""
    return sorted({result.metric for result in results if result.is_scored})
