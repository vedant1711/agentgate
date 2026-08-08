"""Reliability metrics from K repetitions (B6, E2).

tau-bench's central observation: a GPT-4o agent averaging above 60% success drops below 25% at
pass^8. Average success and *reliability* are different quantities, and only the first is
visible in a single-run eval. So AgentGate runs every task K times and reports both.

Estimators are the unbiased combinatorial ones, not plug-in rates:

* ``pass@k = 1 - C(K-c, k)/C(K, k)`` — probability at least one of ``k`` draws succeeds
  (Chen et al., arXiv:2107.03374).
* ``pass^k = C(c, k)/C(K, k)`` — probability **all** ``k`` draws succeed, the same combinatorial
  argument run the other way.

Both are unbiased for the corresponding functional of the per-task success probability, which
the Monte-Carlo suite verifies directly. The naive plug-in ``(c/K)^k`` is biased downward and
the difference is largest exactly where it matters, near p = 1.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from agentgate.schemas.results import (
    Estimate,
    PassKPoint,
    ReliabilityReport,
    TaskFlake,
)
from agentgate.stats.intervals import DEFAULT_BOOTSTRAP, bootstrap_bca, clt_interval

SUCCESS_THRESHOLD = 0.5
"""A repetition counts as a success at or above this score."""


@dataclass(frozen=True, slots=True)
class TaskOutcomes:
    """One task's per-repetition binary outcomes."""

    task_id: str
    successes: int
    trials: int
    cluster_id: str = ""

    @property
    def rate(self) -> float:
        """Empirical success rate."""
        return self.successes / self.trials if self.trials else 0.0

    @property
    def is_flaky(self) -> bool:
        """True when the task succeeded sometimes but not always — the CI-killing behaviour."""
        return 0 < self.successes < self.trials


def pass_at_k(successes: int, trials: int, k: int) -> float:
    """Unbiased estimator of "at least one of k draws succeeds".

    Args:
        successes: Observed successes ``c`` out of ``trials``.
        trials: Repetitions ``K`` actually run.
        k: Draw count to estimate for; must not exceed ``trials``.

    Returns:
        The estimate in [0, 1].

    Raises:
        ValueError: When ``k`` exceeds ``trials`` — extrapolating past the data would be an
            invention, not an estimate.
    """
    _validate(successes, trials, k)
    failures = trials - successes
    if failures < k:
        return 1.0
    return 1.0 - math.comb(failures, k) / math.comb(trials, k)


def pass_hat_k(successes: int, trials: int, k: int) -> float:
    """Unbiased estimator of "**all** k draws succeed" — the pass^k of E2.

    Args:
        successes: Observed successes ``c`` out of ``trials``.
        trials: Repetitions ``K`` actually run.
        k: Draw count to estimate for.

    Returns:
        The estimate in [0, 1]. Note ``pass^1`` equals the plain success rate, and the curve is
        non-increasing in ``k`` by construction.

    Raises:
        ValueError: When ``k`` exceeds ``trials``.
    """
    _validate(successes, trials, k)
    if successes < k:
        return 0.0
    return math.comb(successes, k) / math.comb(trials, k)


def _validate(successes: int, trials: int, k: int) -> None:
    if trials <= 0:
        msg = "reliability estimators need at least one trial"
        raise ValueError(msg)
    if not 0 <= successes <= trials:
        msg = f"successes {successes} out of range for {trials} trials"
        raise ValueError(msg)
    if not 1 <= k <= trials:
        msg = (
            f"k={k} must be between 1 and K={trials}; estimating pass^k beyond the repetitions "
            f"actually run would be extrapolation, not estimation"
        )
        raise ValueError(msg)


def outcomes_from_scores(
    scores: dict[str, list[float]],
    *,
    clusters: dict[str, str] | None = None,
    threshold: float = SUCCESS_THRESHOLD,
) -> list[TaskOutcomes]:
    """Convert per-task repetition scores into success counts.

    Args:
        scores: ``{task_id: [score per repetition]}``.
        clusters: ``{task_id: cluster_id}``.
        threshold: Score at or above which a repetition counts as a success.

    Returns:
        One :class:`TaskOutcomes` per task, in task-id order.
    """
    lookup = clusters or {}
    return [
        TaskOutcomes(
            task_id=task_id,
            successes=sum(1 for value in values if value >= threshold),
            trials=len(values),
            cluster_id=lookup.get(task_id, task_id),
        )
        for task_id, values in sorted(scores.items())
        if values
    ]


def flake_rate(outcomes: Sequence[TaskOutcomes]) -> float:
    """Share of tasks that succeeded sometimes but not always."""
    if not outcomes:
        return 0.0
    return sum(1 for outcome in outcomes if outcome.is_flaky) / len(outcomes)


def score_variance(scores: dict[str, list[float]]) -> float:
    """Mean within-task variance across repetitions.

    Surfaces nondeterminism that suite means hide: two agents can share a mean and differ
    completely in whether any individual task is dependable.
    """
    variances = [float(np.var(values, ddof=1)) for values in scores.values() if len(values) > 1]
    return float(np.mean(variances)) if variances else 0.0


def reliability_report(
    base_metric: str,
    scores: dict[str, list[float]],
    *,
    clusters: dict[str, str] | None = None,
    threshold: float = SUCCESS_THRESHOLD,
    resamples: int = DEFAULT_BOOTSTRAP,
    seed: int = 20260101,
    level: float = 0.95,
) -> ReliabilityReport:
    """Build the full reliability panel for one binary base metric.

    pass^k and flake rate get **BCa bootstrap** intervals rather than CLT ones: both are bounded
    and skewed near their limits, which is precisely where a normal interval misbehaves.

    Args:
        base_metric: Name of the binary metric the reliability curve is built from.
        scores: ``{task_id: [score per repetition]}``.
        clusters: ``{task_id: cluster_id}`` for cluster-level resampling.
        threshold: Success threshold per repetition.
        resamples: Bootstrap replicates.
        seed: RNG seed.
        level: Confidence level.

    Returns:
        The report, with the pass^k curve for ``k = 1..K``.
    """
    outcomes = outcomes_from_scores(scores, clusters=clusters, threshold=threshold)
    if not outcomes:
        return ReliabilityReport(
            base_metric=base_metric,
            k_max=1,
            flake_rate=Estimate(value=0.0, method="empty", n=0),
        )

    k_max = min(outcome.trials for outcome in outcomes)
    cluster_ids = [outcome.cluster_id for outcome in outcomes]
    use_clusters = len(set(cluster_ids)) < len(outcomes)

    curve: list[PassKPoint] = []
    for k in range(1, k_max + 1):
        at_k = [pass_at_k(o.successes, o.trials, k) for o in outcomes]
        hat_k = [pass_hat_k(o.successes, o.trials, k) for o in outcomes]
        curve.append(
            PassKPoint(
                k=k,
                pass_at_k=clt_interval(at_k, level=level, method=f"clt(pass@{k})"),
                pass_hat_k=bootstrap_bca(
                    hat_k,
                    lambda sample: float(np.mean(sample)) if sample else 0.0,
                    clusters=cluster_ids if use_clusters else None,
                    resamples=resamples,
                    level=level,
                    seed=seed + k,
                    method_label=f"bca(pass^{k})",
                ),
            )
        )

    flake_flags = [1.0 if outcome.is_flaky else 0.0 for outcome in outcomes]
    flake = bootstrap_bca(
        flake_flags,
        lambda sample: float(np.mean(sample)) if sample else 0.0,
        clusters=cluster_ids if use_clusters else None,
        resamples=resamples,
        level=level,
        seed=seed,
        method_label="bca(flake_rate)",
    )

    flakiest = sorted(
        (
            TaskFlake(task_id=o.task_id, successes=o.successes, k=o.trials)
            for o in outcomes
            if o.is_flaky
        ),
        key=lambda flake_entry: (abs(flake_entry.rate - 0.5), flake_entry.task_id),
    )

    return ReliabilityReport(
        base_metric=base_metric,
        k_max=k_max,
        curve=curve,
        flake_rate=flake,
        score_variance=score_variance(scores),
        flakiest_tasks=flakiest[:10],
    )


def pass_hat_k_series(outcomes: Sequence[TaskOutcomes], k: int) -> list[float]:
    """Per-task pass^k values at one ``k`` — the analysis unit when the gate targets pass^k."""
    return [pass_hat_k(o.successes, o.trials, k) for o in outcomes if o.trials >= k]


def naive_pass_hat_k(successes: int, trials: int, k: int) -> float:
    """The biased plug-in ``(c/K)^k``, kept only so the report can show the difference."""
    _validate(successes, trials, k)
    return (successes / trials) ** k
