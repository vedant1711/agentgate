"""Paired baseline-vs-candidate comparison (C2), assembled per metric.

This module produces the evidence; the gate engine (Part E) reads it and rules. Keeping those
separate matters: the same comparison must be re-rulable at a different margin, alpha, or FDR
level without re-running anything, which is what makes the interactive demo possible and what
lets a reviewer check whether a verdict survives a different policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from agentgate.errors import InsufficientDataError
from agentgate.schemas.common import Direction
from agentgate.schemas.results import (
    Estimate,
    MetricComparison,
    MetricResult,
    PairedTestResult,
)
from agentgate.stats.aggregate import repetition_scores, task_means
from agentgate.stats.intervals import clip_estimate, clt_interval, clustered_interval
from agentgate.stats.paired import (
    DEFAULT_PERMUTATIONS,
    PairedData,
    cohens_dz,
    difference_estimate,
    disagreement,
    discordant_counts,
    mcnemar_exact,
    mcnemar_odds_ratio,
    paired_correlation,
    permutation_test,
    run_paired_test,
)
from agentgate.stats.power import PowerInputs, power_report
from agentgate.stats.reliability import SUCCESS_THRESHOLD


@dataclass(slots=True)
class ComparisonConfig:
    """Levels and knobs for one paired comparison."""

    alpha: float = 0.05
    level: float = 0.95
    normality_alpha: float = 0.05
    permutation_iters: int = DEFAULT_PERMUTATIONS
    power_target: float = 0.80
    use_clusters: bool = True
    seed: int = 20260101
    run_permutation: bool = True
    margins: dict[str, float] = field(default_factory=dict)
    """``{metric: absolute margin}``. Metrics absent here are compared at margin 0."""


def build_paired_data(
    baseline: Sequence[MetricResult],
    candidate: Sequence[MetricResult],
    *,
    clusters: dict[str, str] | None = None,
) -> PairedData:
    """Align two runs' per-sample results into per-task paired scores.

    Only tasks scored in **both** runs are kept. Silently comparing a task one side skipped
    would compare a score against nothing.

    Args:
        baseline: Baseline per-sample results for one metric.
        candidate: Candidate per-sample results for the same metric.
        clusters: ``{task_id: cluster_id}``.

    Returns:
        The aligned paired data.

    Raises:
        InsufficientDataError: When no task was scored on both sides.
    """
    base_means = task_means(repetition_scores(baseline))
    cand_means = task_means(repetition_scores(candidate))
    shared = sorted(set(base_means) & set(cand_means))
    if not shared:
        msg = "no task was scored in both runs; there is nothing to pair"
        raise InsufficientDataError(msg)
    lookup = clusters or {}
    return PairedData(
        task_ids=tuple(shared),
        baseline=tuple(base_means[task_id] for task_id in shared),
        candidate=tuple(cand_means[task_id] for task_id in shared),
        clusters=tuple(lookup.get(task_id, task_id) for task_id in shared),
    )


def cluster_mean_differences(data: PairedData) -> list[float]:
    """Per-cluster mean differences — the independent units when a suite declares clusters.

    Args:
        data: Direction-normalised paired data.

    Returns:
        One difference per cluster, in cluster-id order. Falls back to per-task differences when
        no clustering is present.
    """
    if not data.has_clusters:
        return data.differences
    groups: dict[str, list[float]] = {}
    for value, cluster in zip(data.differences, data.clusters, strict=True):
        groups.setdefault(cluster, []).append(value)
    return [float(np.mean(groups[cluster])) for cluster in sorted(groups)]


def _side_estimate(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    level: float,
    use_clusters: bool,
) -> Estimate:
    """One system's suite mean, cluster-robust when the design warrants it."""
    if use_clusters and len(set(clusters)) < len(clusters):
        return clustered_interval(values, clusters, level=level)
    return clt_interval(values, level=level)


def compare_metric(
    metric: str,
    baseline: Sequence[MetricResult],
    candidate: Sequence[MetricResult],
    *,
    clusters: dict[str, str] | None = None,
    margin: float = 0.0,
    config: ComparisonConfig | None = None,
) -> MetricComparison:
    """Run the full paired analysis for one metric.

    Both one-sided tests of C3 are computed here:

    * **regression** — ``H0: Delta >= -delta`` against ``H1: Delta < -delta``. Rejecting means
      the candidate is worse by more than the margin.
    * **non-inferiority** — ``H0: Delta <= -delta`` against ``H1: Delta > -delta``. Rejecting
      means the candidate is proven fine.

    Both operate on the shifted differences ``d_i + delta``, and both can fail to reject — which
    is the ``INCONCLUSIVE`` case the gate must be able to express and a naive threshold cannot.

    Differences are direction-normalised first, so a latency regression and a success regression
    are the same arithmetic.

    Args:
        metric: Metric name.
        baseline: Baseline per-sample results.
        candidate: Candidate per-sample results.
        clusters: ``{task_id: cluster_id}``.
        margin: Non-inferiority margin, always non-negative.
        config: Levels and knobs.

    Returns:
        The comparison, with no verdict — the gate assigns that.

    Raises:
        InsufficientDataError: When the two runs share no scored task.
    """
    settings = config or ComparisonConfig()
    template = (baseline or candidate)[0]
    direction: Direction = template.direction

    raw = build_paired_data(baseline, candidate, clusters=clusters)
    data = raw.directed(direction)
    n = data.n
    clustered = settings.use_clusters and data.has_clusters

    # When tasks are clustered, the *tests* analyse cluster mean differences, not raw per-task
    # differences. Testing at the task level while reporting a cluster-robust interval would be
    # incoherent — and anti-conservative in exactly the direction E3 warns about, since the
    # task-level test would treat five paraphrases of one scenario as five independent facts.
    differences = cluster_mean_differences(data) if clustered else data.differences

    baseline_estimate = _side_estimate(
        raw.baseline, raw.clusters, level=settings.level, use_clusters=clustered
    )
    candidate_estimate = _side_estimate(
        raw.candidate, raw.clusters, level=settings.level, use_clusters=clustered
    )
    if template.dtype in ("binary", "proportion"):
        baseline_estimate = clip_estimate(baseline_estimate)
        candidate_estimate = clip_estimate(candidate_estimate)
    delta = difference_estimate(data, level=settings.level, use_clusters=clustered)

    notes: list[str] = []
    if clustered:
        notes.append(
            f"clustered analysis: {n} tasks in {len(differences)} clusters. Tests and intervals "
            f"both use per-cluster mean differences, so five paraphrases of one scenario count "
            f"as one independent observation, not five (C1.3)."
        )
    if n < 2:
        empty = PairedTestResult(test="none", p_one_sided=1.0, n_pairs=n)
        notes.append(f"only {n} paired task(s); no test is possible")
        return MetricComparison(
            metric=metric,
            family=template.family,
            dtype=template.dtype,
            direction=direction,
            n_pairs=n,
            baseline=baseline_estimate,
            candidate=candidate_estimate,
            delta=delta,
            margin=margin,
            regression_test=empty,
            noninferiority_test=empty,
            clustered=clustered,
            notes=notes,
        )

    regression = run_paired_test(
        differences,
        shift=margin,
        alternative="less",
        normality_alpha=settings.normality_alpha,
    )
    noninferiority = run_paired_test(
        differences,
        shift=margin,
        alternative="greater",
        normality_alpha=settings.normality_alpha,
    )

    permutation = None
    if settings.run_permutation:
        permutation = permutation_test(
            differences,
            shift=margin,
            alternative="less",
            iterations=settings.permutation_iters,
            seed=settings.seed,
        )
        flagged = disagreement(regression, permutation)
        if flagged:
            notes.append(flagged)

    effect = cohens_dz(differences)
    if template.dtype == "binary":
        b, c = discordant_counts(data, threshold=SUCCESS_THRESHOLD)
        mcnemar = mcnemar_exact(b, c)
        notes.append(
            f"McNemar exact on discordant pairs (b={b}, c={c}): "
            f"p={mcnemar.p_one_sided:.4f} at margin 0. The gate's test uses the shifted paired "
            f"differences, which is what carries the margin (C3)."
        )
        odds = mcnemar_odds_ratio(b, c)
        if odds is not None:
            effect = odds

    sigma_d = float(np.std(differences, ddof=1)) if n >= 2 else 0.0
    power = power_report(
        PowerInputs(
            sigma_d=sigma_d,
            n_pairs=delta.n,
            margin=margin,
            alpha=settings.alpha,
            target=settings.power_target,
        )
    )

    correlation = paired_correlation(data)
    if correlation is not None and correlation < 0.1:
        notes.append(
            f"paired correlation r={correlation:.2f} is low; pairing bought little variance "
            f"reduction here, so this metric needs close to the unpaired sample size"
        )

    return MetricComparison(
        metric=metric,
        family=template.family,
        dtype=template.dtype,
        direction=direction,
        n_pairs=n,
        baseline=baseline_estimate,
        candidate=candidate_estimate,
        delta=delta,
        margin=margin,
        analysis_units=list(differences),
        correlation=correlation,
        effect_size=effect,
        regression_test=regression,
        noninferiority_test=noninferiority,
        permutation_test=permutation,
        power=power,
        clustered=clustered,
        notes=notes,
    )


def compare_all(
    baseline: Sequence[MetricResult],
    candidate: Sequence[MetricResult],
    *,
    metrics: Sequence[str] | None = None,
    clusters: dict[str, str] | None = None,
    config: ComparisonConfig | None = None,
) -> list[MetricComparison]:
    """Compare every metric present in both runs.

    Args:
        baseline: All baseline per-sample results.
        candidate: All candidate per-sample results.
        metrics: Restrict to these metric names; ``None`` compares everything comparable.
        clusters: ``{task_id: cluster_id}``.
        config: Levels and knobs, including per-metric margins.

    Returns:
        One comparison per metric, in name order. Metrics that cannot be paired are skipped
        rather than reported with an invented result.
    """
    settings = config or ComparisonConfig()
    base_by_metric: dict[str, list[MetricResult]] = {}
    cand_by_metric: dict[str, list[MetricResult]] = {}
    for result in baseline:
        base_by_metric.setdefault(result.metric, []).append(result)
    for result in candidate:
        cand_by_metric.setdefault(result.metric, []).append(result)

    wanted = (
        sorted(set(metrics))
        if metrics is not None
        else sorted(set(base_by_metric) & set(cand_by_metric))
    )
    comparisons: list[MetricComparison] = []
    for metric in wanted:
        if metric not in base_by_metric or metric not in cand_by_metric:
            continue
        try:
            comparisons.append(
                compare_metric(
                    metric,
                    base_by_metric[metric],
                    cand_by_metric[metric],
                    clusters=clusters,
                    margin=settings.margins.get(metric, 0.0),
                    config=settings,
                )
            )
        except InsufficientDataError:
            continue
    return comparisons
