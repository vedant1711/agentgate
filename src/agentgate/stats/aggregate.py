"""Single-run aggregation (C1): per-sample scores to metric summaries with uncertainty.

The analysis unit is the **task**, never the repetition. Repetitions of the same task are not
independent draws from the task distribution — treating them as such would shrink every interval
by a factor of sqrt(K) for nothing. So each task's K repetitions are averaged first, and the
suite statistics run over the resulting per-task scores.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from agentgate.schemas.results import (
    Estimate,
    MetricResult,
    MetricSummary,
    RunManifest,
    RunReport,
)
from agentgate.stats.intervals import (
    clip_estimate,
    clt_interval,
    clustered_interval,
    se_inflation,
    wilson_interval,
)
from agentgate.stats.reliability import SUCCESS_THRESHOLD, reliability_report
from agentgate.stats.variance import decompose, inflate_estimate

RELIABILITY_BASE_METRICS = ("outcome.task_success",)
"""Binary metrics that get a full pass^k reliability panel."""


def repetition_scores(results: Iterable[MetricResult]) -> dict[str, list[float]]:
    """Group usable per-sample values by task, in repetition order.

    Skipped and errored samples are excluded rather than imputed: a metric that could not be
    scored contributes nothing, and inventing a value for it would move the suite mean.
    """
    grouped: dict[str, list[tuple[int, float]]] = {}
    for result in results:
        if not result.is_scored or result.value is None:
            continue
        grouped.setdefault(result.task_id, []).append((result.rep, result.value))
    return {
        task_id: [value for _, value in sorted(pairs)] for task_id, pairs in sorted(grouped.items())
    }


def task_means(scores: dict[str, list[float]]) -> dict[str, float]:
    """Mean score per task — the analysis unit for every suite-level statistic."""
    return {task_id: float(np.mean(values)) for task_id, values in scores.items() if values}


def judge_variances(results: Iterable[MetricResult]) -> list[float]:
    """Per-sample judge variances, for the C5 propagation."""
    return [
        result.judge_variance
        for result in results
        if result.judge_variance is not None and result.is_scored
    ]


def summarise_metric(
    results: Sequence[MetricResult],
    *,
    clusters: dict[str, str] | None = None,
    level: float = 0.95,
    n_judge_samples: int = 3,
    use_clusters: bool = True,
) -> MetricSummary | None:
    """Aggregate one metric's per-sample results into a summary with uncertainty.

    Interval choice follows the measurement type and the design:

    * **binary at K=1** — Wilson. Each task is a single Bernoulli trial, which is exactly the
      regime Wilson is built for and where the normal approximation is worst.
    * **binary at K>1** — CLT on per-task success *rates*. A rate averaged over repetitions is
      not Bernoulli, so Wilson's variance formula no longer applies; the method label records
      which was used.
    * **everything else** — CLT on per-task means.

    When the suite declares clusters, a cluster-robust interval is computed alongside and the
    inflation ratio is reported, so the reader sees the difference rather than being told it.

    Args:
        results: All per-sample results for one metric.
        clusters: ``{task_id: cluster_id}``.
        level: Confidence level.
        n_judge_samples: J, for judge-variance propagation.
        use_clusters: Whether to compute the clustered interval.

    Returns:
        The summary, or ``None`` when no sample was scoreable.
    """
    if not results:
        return None
    scored = [result for result in results if result.is_scored]
    skipped = len(results) - len(scored)
    if not scored:
        return None

    first = scored[0]
    scores = repetition_scores(scored)
    means = task_means(scores)
    if not means:
        return None

    task_ids = sorted(means)
    values = [means[task_id] for task_id in task_ids]
    k = max(len(reps) for reps in scores.values())

    if first.dtype == "binary" and k == 1:
        successes = sum(1 for value in values if value >= SUCCESS_THRESHOLD)
        estimate = wilson_interval(successes, len(values), level=level)
    else:
        method = "clt(task-rates)" if first.dtype == "binary" else "clt"
        estimate = clt_interval(values, level=level, method=method)
    if first.dtype in ("binary", "proportion"):
        estimate = clip_estimate(estimate)

    clustered = None
    inflation = None
    if use_clusters and clusters:
        cluster_ids = [clusters.get(task_id, task_id) for task_id in task_ids]
        if len(set(cluster_ids)) < len(cluster_ids):
            clustered = clustered_interval(values, cluster_ids, level=level)
            if first.dtype in ("binary", "proportion"):
                clustered = clip_estimate(clustered)
            inflation = se_inflation(estimate, clustered)

    variances = judge_variances(scored)
    if variances:
        estimate = inflate_estimate(
            estimate,
            variances,
            n_judge_samples=n_judge_samples,
            n_reps=k,
            task_means=values,
        )

    return MetricSummary(
        metric=first.metric,
        family=first.family,
        dtype=first.dtype,
        direction=first.direction,
        n_tasks=len(values),
        n_scored_samples=len(scored),
        n_skipped=skipped,
        k=k,
        estimate=estimate,
        clustered=clustered,
        se_inflation=inflation,
        variance=decompose(scores),
        per_task=means,
        judge_backed=bool(variances) or first.metric.startswith("judge."),
    )


def summarise_run(
    manifest: RunManifest,
    results: Iterable[MetricResult],
    *,
    clusters: dict[str, str] | None = None,
    level: float = 0.95,
    reliability_metrics: Sequence[str] = RELIABILITY_BASE_METRICS,
    bootstrap_resamples: int = 2_000,
) -> RunReport:
    """Build the full single-run analysis.

    Args:
        manifest: The run's manifest.
        results: Every per-sample metric result.
        clusters: ``{task_id: cluster_id}``.
        level: Confidence level.
        reliability_metrics: Binary metrics to build pass^k panels from.
        bootstrap_resamples: Replicates for the reliability BCa intervals. Lower than the C1
            default of 10,000 because this runs per metric per run; the gate path uses the full
            count.

    Returns:
        The report, with a warning for any metric that could not be scored at all.
    """
    by_metric: dict[str, list[MetricResult]] = {}
    for result in results:
        by_metric.setdefault(result.metric, []).append(result)

    summaries: list[MetricSummary] = []
    warnings: list[str] = []
    for metric in sorted(by_metric):
        summary = summarise_metric(
            by_metric[metric], clusters=clusters, level=level, n_judge_samples=manifest.k
        )
        if summary is None:
            warnings.append(f"{metric}: no sample was scoreable in this run")
            continue
        summaries.append(summary)
        if summary.n_tasks < 5:
            warnings.append(
                f"{metric}: only {summary.n_tasks} task(s) scored; intervals are very wide"
            )

    reliability = []
    for metric in reliability_metrics:
        if metric not in by_metric:
            continue
        scores = repetition_scores(by_metric[metric])
        if not scores:
            continue
        reliability.append(
            reliability_report(
                metric,
                scores,
                clusters=clusters,
                resamples=bootstrap_resamples,
                seed=manifest.base_seed,
                level=level,
            )
        )

    return RunReport(
        run_id=manifest.run_id,
        manifest=manifest,
        summaries=summaries,
        reliability=reliability,
        warnings=warnings,
    )


def latency_percentiles(
    results: Sequence[MetricResult], *, level: float = 0.95
) -> dict[str, Estimate]:
    """p50 and p95 latency with BCa intervals.

    A mean latency hides the tail that actually makes an agent unusable, so the efficiency panel
    reports quantiles — and bootstraps them, because a quantile has no clean closed-form SE.
    """
    from agentgate.stats.intervals import bootstrap_bca

    values = [result.value for result in results if result.is_scored and result.value is not None]
    if not values:
        return {}
    return {
        "p50": bootstrap_bca(
            values,
            lambda sample: float(np.quantile(sample, 0.5)),
            resamples=2_000,
            level=level,
            method_label="bca(p50)",
        ),
        "p95": bootstrap_bca(
            values,
            lambda sample: float(np.quantile(sample, 0.95)),
            resamples=2_000,
            level=level,
            method_label="bca(p95)",
        ),
    }
