"""Confidence intervals (C1).

*An eval score is an estimate of a population mean over a task distribution, so it must ship
with uncertainty* (Miller, arXiv:2411.00640). Four procedures, each with a regime it is right
for:

* :func:`clt_interval` — the workhorse for continuous and proportion metrics.
* :func:`wilson_interval` — binary metrics. The normal approximation has genuinely bad coverage
  at small ``n`` and extreme ``p``, and portfolio-scale suites are exactly that regime
  (n = 30-200, p often above 0.9). Wilson (1927) fixes it.
* :func:`clustered_interval` — when tasks come in related groups. Naive SEs can understate
  uncertainty by more than 3x here, which is the difference between a real finding and noise.
* :func:`bootstrap_bca` — for statistics with no clean sampling distribution: p95 latency,
  pass^k, flake rate.

Every function returns an :class:`~agentgate.schemas.results.Estimate`, which cannot be built
without naming the method that produced its interval.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats

from agentgate.errors import InsufficientDataError
from agentgate.schemas.results import Estimate

DEFAULT_LEVEL = 0.95
DEFAULT_BOOTSTRAP = 10_000


def z_for(level: float = DEFAULT_LEVEL) -> float:
    """Two-sided normal critical value for a confidence level."""
    return float(stats.norm.ppf(0.5 + level / 2.0))


def clt_interval(
    values: Sequence[float], *, level: float = DEFAULT_LEVEL, method: str = "clt"
) -> Estimate:
    """Mean with a CLT standard error and normal interval.

    Args:
        values: Independent observations — **task-level** scores, already averaged over the K
            repetitions. Repetitions are not independent tasks, and treating them as such would
            shrink the interval by a factor of sqrt(K) for free.
        level: Confidence level.
        method: Label recorded on the estimate.

    Returns:
        The estimate. With ``n < 2`` the SE is undefined and the estimate says so rather than
        reporting a zero-width interval.
    """
    data = np.asarray(values, dtype=float)
    n = int(data.size)
    if n == 0:
        return Estimate(value=float("nan"), method=f"{method}(empty)", n=0)
    mean = float(data.mean())
    if n < 2:
        return Estimate(value=mean, method=f"{method}(n<2)", n=n)
    se = float(data.std(ddof=1) / math.sqrt(n))
    half = z_for(level) * se
    return Estimate(
        value=mean,
        se=se,
        ci_low=mean - half,
        ci_high=mean + half,
        ci_level=level,
        method=method,
        n=n,
    )


def wilson_interval(successes: float, trials: int, *, level: float = DEFAULT_LEVEL) -> Estimate:
    """Wilson score interval for a binomial proportion (Wilson, 1927).

    The interval is the set of ``p`` the score test does not reject, which keeps it inside
    [0, 1] and gives it far better small-sample coverage than ``p +/- z*sqrt(p(1-p)/n)`` — an
    approximation that produces intervals extending past 1.0 whenever ``p`` is near the boundary,
    exactly where a good agent lives.

    Args:
        successes: Number of successes (may be fractional for averaged repetitions).
        trials: Number of independent trials.
        level: Confidence level.

    Returns:
        The estimate.

    Raises:
        InsufficientDataError: When ``trials`` is not positive.
    """
    if trials <= 0:
        msg = "Wilson interval needs at least one trial"
        raise InsufficientDataError(msg)
    z = z_for(level)
    n = float(trials)
    p_hat = successes / n
    denominator = 1.0 + z * z / n
    centre = (p_hat + z * z / (2.0 * n)) / denominator
    half = (z / denominator) * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
    se = math.sqrt(p_hat * (1.0 - p_hat) / n) if 0.0 < p_hat < 1.0 else 0.0
    # At p_hat = 0 or 1 the algebra puts the bound exactly on the boundary, but floating point
    # lands an ULP short and the interval then fails to contain its own point estimate. Clamping
    # to include p_hat keeps every downstream comparison against a bound coherent.
    return Estimate(
        value=p_hat,
        se=se,
        ci_low=max(0.0, min(p_hat, centre - half)),
        ci_high=min(1.0, max(p_hat, centre + half)),
        ci_level=level,
        method="wilson",
        n=trials,
    )


def clustered_interval(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    level: float = DEFAULT_LEVEL,
) -> Estimate:
    """Cluster-robust interval for a mean (C1.3).

    Uses the sandwich estimator for the sample mean with a CR1 finite-sample correction:

    ``V = G/(G-1) * sum_g (sum_i (y_gi - ybar))^2 / n^2``

    When tasks are paraphrases of the same scenario, their errors are correlated; the naive SE
    assumes they are not, and reports an interval that is too narrow by however much that
    assumption is wrong. The report shows both so the reader can see the difference rather than
    take it on trust.

    Args:
        values: Task-level scores.
        clusters: Cluster id per value, aligned.
        level: Confidence level.

    Returns:
        The estimate, with ``n`` set to the number of **clusters** — the true number of
        independent units, and the honest denominator for this design.

    Raises:
        InsufficientDataError: When the inputs are misaligned.
    """
    if len(values) != len(clusters):
        msg = f"clustered interval needs aligned inputs: {len(values)} values, {len(clusters)} ids"
        raise InsufficientDataError(msg)
    data = np.asarray(values, dtype=float)
    n = int(data.size)
    if n == 0:
        return Estimate(value=float("nan"), method="clustered(empty)", n=0)

    mean = float(data.mean())
    groups: dict[str, list[float]] = {}
    for value, cluster in zip(data.tolist(), clusters, strict=True):
        groups.setdefault(cluster, []).append(value)
    n_clusters = len(groups)
    if n_clusters < 2:
        return Estimate(value=mean, method="clustered(G<2)", n=n_clusters)

    residual_sums = [sum(value - mean for value in group) for group in groups.values()]
    variance = (n_clusters / (n_clusters - 1.0)) * sum(r * r for r in residual_sums) / (n * n)
    se = math.sqrt(max(0.0, variance))
    half = z_for(level) * se
    return Estimate(
        value=mean,
        se=se,
        ci_low=mean - half,
        ci_high=mean + half,
        ci_level=level,
        method="clustered",
        n=n_clusters,
    )


def se_inflation(naive: Estimate, clustered: Estimate) -> float | None:
    """Ratio of clustered to naive standard error — the number E3 says can exceed 3."""
    if naive.se in (None, 0.0) or clustered.se is None:
        return None
    return clustered.se / naive.se


def percentile_interval(
    replicates: Sequence[float], *, level: float = DEFAULT_LEVEL
) -> tuple[float, float]:
    """Plain percentile bounds from a bootstrap distribution."""
    alpha = (1.0 - level) / 2.0
    array = np.asarray(replicates, dtype=float)
    return float(np.quantile(array, alpha)), float(np.quantile(array, 1.0 - alpha))


def bootstrap_bca(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    *,
    clusters: Sequence[str] | None = None,
    resamples: int = DEFAULT_BOOTSTRAP,
    level: float = DEFAULT_LEVEL,
    seed: int = 20260101,
    method_label: str = "bca",
) -> Estimate:
    """Bias-corrected and accelerated bootstrap interval (Efron).

    BCa corrects two things the plain percentile interval gets wrong: **bias** (the bootstrap
    distribution not centred on the observed statistic, via ``z0``) and **skew** (the standard
    error varying with the parameter, via the jackknife acceleration ``a``). Both matter for the
    statistics this is used on — pass^k is a bounded, skewed quantity near 1.0, and p95 latency
    is a tail estimate.

    Resampling is done **at the cluster level** when clusters exist, because resampling
    individual tasks would break the very dependence the clustering represents.

    Args:
        values: Observations.
        statistic: Function computing the statistic from a sample.
        clusters: Cluster id per value, when the design is clustered.
        resamples: Bootstrap replicates.
        level: Confidence level.
        seed: RNG seed, so an interval is reproducible.
        method_label: Label recorded on the estimate.

    Returns:
        The estimate. Falls back to the percentile interval when the acceleration is undefined
        (a degenerate jackknife), recording that in the method label.
    """
    data = list(values)
    n = len(data)
    if n == 0:
        return Estimate(value=float("nan"), method=f"{method_label}(empty)", n=0)
    observed = float(statistic(data))
    if n < 3:
        return Estimate(value=observed, method=f"{method_label}(n<3)", n=n)

    rng = np.random.default_rng(seed)
    if clusters is not None:
        groups: dict[str, list[float]] = {}
        for value, cluster in zip(data, clusters, strict=True):
            groups.setdefault(cluster, []).append(value)
        keys = sorted(groups)
        replicates = np.empty(resamples, dtype=float)
        for index in range(resamples):
            picked = rng.integers(0, len(keys), size=len(keys))
            sample = [value for position in picked for value in groups[keys[position]]]
            replicates[index] = statistic(sample)
        units = len(keys)
        jackknife = np.array(
            [statistic([v for key in keys if key != drop for v in groups[key]]) for drop in keys]
        )
    else:
        array = np.asarray(data, dtype=float)
        picks = rng.integers(0, n, size=(resamples, n))
        replicates = np.array([statistic(array[row].tolist()) for row in picks], dtype=float)
        units = n
        jackknife = np.array(
            [statistic(np.delete(array, index).tolist()) for index in range(n)], dtype=float
        )

    proportion_below = float(np.mean(replicates < observed))
    if proportion_below <= 0.0 or proportion_below >= 1.0:
        low, high = percentile_interval(replicates.tolist(), level=level)
        return Estimate(
            value=observed,
            se=float(replicates.std(ddof=1)),
            ci_low=low,
            ci_high=high,
            ci_level=level,
            method=f"{method_label}->percentile(degenerate z0)",
            n=units,
        )

    z0 = float(stats.norm.ppf(proportion_below))
    jack_mean = float(jackknife.mean())
    deviations = jack_mean - jackknife
    denominator = 6.0 * float(np.sum(deviations**2)) ** 1.5
    acceleration = float(np.sum(deviations**3)) / denominator if denominator > 0 else 0.0

    alpha = (1.0 - level) / 2.0
    z_low, z_high = float(stats.norm.ppf(alpha)), float(stats.norm.ppf(1.0 - alpha))
    adjusted_low = _adjust(z0, acceleration, z_low)
    adjusted_high = _adjust(z0, acceleration, z_high)

    low = float(np.quantile(replicates, adjusted_low))
    high = float(np.quantile(replicates, adjusted_high))
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        low, high = percentile_interval(replicates.tolist(), level=level)
        method_label = f"{method_label}->percentile(unstable)"
    return Estimate(
        value=observed,
        se=float(replicates.std(ddof=1)),
        ci_low=low,
        ci_high=high,
        ci_level=level,
        method=method_label,
        n=units,
    )


def _adjust(z0: float, acceleration: float, z: float) -> float:
    """BCa's percentile adjustment, clipped into a usable quantile range."""
    numerator = z0 + z
    denominator = 1.0 - acceleration * numerator
    if denominator == 0.0:
        return float(stats.norm.cdf(z0 + numerator))
    return float(min(0.999999, max(0.000001, stats.norm.cdf(z0 + numerator / denominator))))


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile, used for p50/p95 latency reporting."""
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=float), q))
