"""Variance decomposition and judge-uncertainty propagation (C1.5, C5).

Two questions this module answers.

**"More tasks or more repetitions?"** A one-way random-effects decomposition splits total
variance into between-task and within-task components. The intraclass correlation is the share
that is between-task, and it answers the budgeting question directly: high ICC means the
variation is real task difficulty and more *tasks* buy precision; low ICC means the agent is
inconsistent on the tasks you already have and more *repetitions* buy precision. On a free tier
that is the only optimisation question that matters.

**"How much of this interval is the judge's fault?"** For judge-backed metrics, the score is a
measurement with error. The law of total variance says

``Var(X) = E[Var(X | item)] + Var(E[X | item])``

so the judge's within-item variance adds to the sampling variance rather than hiding inside it.
Ignoring it reports an interval that is too narrow by exactly the amount the judge is unsure.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from agentgate.schemas.results import Estimate, VarianceDecomposition

ICC_HIGH = 0.7
ICC_LOW = 0.3


def decompose(scores: dict[str, list[float]]) -> VarianceDecomposition | None:
    """One-way random-effects variance decomposition across tasks and repetitions.

    ``MSB = K * sum_i (ybar_i - ybar)^2 / (n - 1)`` and ``MSW = sum_i sum_j (y_ij - ybar_i)^2 /
    (n * (K - 1))`` give ``sigma^2_within = MSW`` and ``sigma^2_between = (MSB - MSW) / K``.

    The between-task component is clipped at zero. A negative estimate is possible when the true
    between-task variance is near zero, and it means "indistinguishable from zero", not "less
    than none".

    Args:
        scores: ``{task_id: [score per repetition]}``, balanced or not — unbalanced designs use
            the mean repetition count, which is exact when balanced and a good approximation
            otherwise.

    Returns:
        The decomposition, or ``None`` when there are too few tasks or repetitions to separate
        the components at all.
    """
    usable = {task: values for task, values in scores.items() if len(values) >= 2}
    n = len(usable)
    if n < 2:
        return None
    reps = [len(values) for values in usable.values()]
    k = float(np.mean(reps))
    if k < 2:
        return None

    task_means = np.array([float(np.mean(values)) for values in usable.values()])
    grand_mean = float(np.mean([value for values in usable.values() for value in values]))

    ms_between = float(k * np.sum((task_means - grand_mean) ** 2) / (n - 1))
    within_sum = sum(
        float(np.sum((np.asarray(values, dtype=float) - np.mean(values)) ** 2))
        for values in usable.values()
    )
    within_df = sum(len(values) - 1 for values in usable.values())
    ms_within = within_sum / within_df if within_df else 0.0

    between_var = max(0.0, (ms_between - ms_within) / k)
    total = between_var + ms_within
    icc = between_var / total if total > 0 else 0.0

    return VarianceDecomposition(
        between_task_var=between_var,
        within_task_var=ms_within,
        icc=icc,
        n_tasks=n,
        k=round(k),
        recommendation=budget_advice(icc),
    )


def budget_advice(icc: float) -> str:
    """Turn an ICC into the sentence a suite author can act on."""
    if icc >= ICC_HIGH:
        return (
            f"ICC={icc:.2f}: most variance is between tasks, so the agent is consistent and the "
            f"suite is the limit. Spend the next quota on more tasks, not more repetitions."
        )
    if icc <= ICC_LOW:
        return (
            f"ICC={icc:.2f}: most variance is within tasks — the agent gives different answers "
            f"to the same task. More repetitions will tighten estimates; more tasks will not. "
            f"This is also a finding in itself: check reliability.flake_rate."
        )
    return (
        f"ICC={icc:.2f}: variance is split between task difficulty and agent inconsistency. "
        f"Additional tasks and additional repetitions buy roughly comparable precision."
    )


def propagate_judge_variance(
    task_means: Sequence[float],
    judge_variances: Sequence[float],
    *,
    n_judge_samples: int,
    n_reps: int = 1,
) -> float:
    """Standard error of a judge-backed metric mean, including judge measurement error (C5).

    ``SE^2 = Var(task means)/n + mean(judge variance) / (J * K * n)``

    The first term is ordinary sampling variance over tasks. The second is the judge's own
    uncertainty: each item's score is a mean of ``J`` draws, each task averages ``K``
    repetitions, and the suite averages ``n`` tasks, so the judge's per-draw variance is damped
    by all three before it reaches the suite mean — but it does not vanish.

    Args:
        task_means: Per-task mean scores.
        judge_variances: Per-item within-judge variance, aligned or aggregated.
        n_judge_samples: J, the number of judge draws per item.
        n_reps: K, repetitions per task.

    Returns:
        The inflated standard error. Falls back to the plain sampling SE when no judge variance
        is available.
    """
    values = np.asarray(task_means, dtype=float)
    n = int(values.size)
    if n < 2:
        return 0.0
    sampling_variance = float(values.var(ddof=1)) / n
    if not judge_variances or n_judge_samples < 1:
        return math.sqrt(sampling_variance)
    mean_judge_variance = float(np.mean(np.asarray(judge_variances, dtype=float)))
    judge_term = mean_judge_variance / (n_judge_samples * max(1, n_reps) * n)
    return math.sqrt(sampling_variance + judge_term)


def inflate_estimate(
    estimate: Estimate,
    judge_variances: Sequence[float],
    *,
    n_judge_samples: int,
    n_reps: int = 1,
    task_means: Sequence[float],
) -> Estimate:
    """Widen an estimate's interval to include judge measurement error (C5).

    Args:
        estimate: The uninflated estimate.
        judge_variances: Per-item judge variance.
        n_judge_samples: J.
        n_reps: K.
        task_means: Per-task means, needed to recompute the sampling term.

    Returns:
        A new estimate whose method label records that judge variance was folded in. Returns the
        input unchanged when there is no judge variance to add.
    """
    if not judge_variances or estimate.se is None:
        return estimate
    inflated = propagate_judge_variance(
        task_means, judge_variances, n_judge_samples=n_judge_samples, n_reps=n_reps
    )
    if inflated <= estimate.se:
        return estimate
    from agentgate.stats.intervals import z_for

    half = z_for(estimate.ci_level) * inflated
    return estimate.model_copy(
        update={
            "se": inflated,
            "ci_low": estimate.value - half,
            "ci_high": estimate.value + half,
            "method": f"{estimate.method}+judge-variance",
        }
    )


def judge_variance_share(sampling_se: float, inflated_se: float) -> float:
    """Share of the total variance attributable to the judge rather than to task sampling.

    Reported next to judge-backed gates: when the judge contributes most of the width, the fix
    is a better judge or a larger J, not a larger suite.
    """
    if inflated_se <= 0.0:
        return 0.0
    total = inflated_se**2
    sampling = min(total, sampling_se**2)
    return float((total - sampling) / total)
