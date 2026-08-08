"""Power analysis and minimum detectable effect (C4).

The honesty machinery. Every gate report answers, in one sentence, *what change this suite could
have detected* — because "no significant regression" from an underpowered suite is not evidence
of no regression, and a gate that cannot tell the difference is worse than no gate.

Formulas are for the **paired** design, where the relevant dispersion is ``sigma_d``, the SD of
the per-task differences — not the SD of the scores. Pairing shrinks ``sigma_d`` by the paired
correlation, which is exactly why the paired design needs fewer tasks for the same power.

The non-central t distribution is used rather than the normal approximation: at n = 30 the
normal approximation overstates power by several points, and overstating power is the specific
failure this module exists to prevent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

from agentgate.schemas.results import PowerReport

MAX_N = 100_000
DEFAULT_POWER = 0.80


def paired_power(
    *, effect: float, sigma_d: float, n: int, alpha: float = 0.05, two_sided: bool = False
) -> float:
    """Power of a paired t-test to detect ``effect``.

    Args:
        effect: True mean difference to detect, in metric units.
        sigma_d: SD of the per-task differences.
        n: Number of paired tasks.
        alpha: Significance level.
        two_sided: Whether the test is two-sided. The gate's tests are one-sided.

    Returns:
        Power in [0, 1]. Returns 1.0 when ``sigma_d`` is 0 and the effect is non-zero (a
        noiseless design detects any real effect), and ``alpha`` when the effect is 0.
    """
    if n < 2:
        return 0.0
    if sigma_d <= 0.0:
        return 1.0 if abs(effect) > 0.0 else alpha
    df = n - 1
    non_centrality = abs(effect) * math.sqrt(n) / sigma_d
    level = alpha / 2.0 if two_sided else alpha
    critical = float(stats.t.ppf(1.0 - level, df))
    power = float(stats.nct.sf(critical, df, non_centrality))
    if two_sided:
        power += float(stats.nct.cdf(-critical, df, non_centrality))
    return float(min(1.0, max(0.0, power)))


def minimum_detectable_effect(
    *, sigma_d: float, n: int, alpha: float = 0.05, power: float = DEFAULT_POWER
) -> float:
    """Smallest effect this design can detect at the target power.

    Solved numerically against :func:`paired_power` rather than via the normal approximation,
    so the answer is the effect the *actual* test would detect.

    Args:
        sigma_d: SD of the per-task differences.
        n: Number of paired tasks.
        alpha: Significance level.
        power: Target power.

    Returns:
        The MDE in metric units; ``inf`` when no effect is detectable at this ``n``.
    """
    if n < 2 or sigma_d <= 0.0:
        return 0.0
    df = n - 1
    # Normal-approximation seed, then bisection on the exact non-central t power curve.
    seed = (
        (float(stats.norm.ppf(1.0 - alpha)) + float(stats.norm.ppf(power))) * sigma_d / math.sqrt(n)
    )
    low, high = 0.0, max(seed * 4.0, sigma_d)
    for _ in range(200):
        if paired_power(effect=high, sigma_d=sigma_d, n=n, alpha=alpha) >= power:
            break
        high *= 2.0
        if high > sigma_d * 1e6:
            return float("inf")
    for _ in range(120):
        mid = (low + high) / 2.0
        if paired_power(effect=mid, sigma_d=sigma_d, n=n, alpha=alpha) >= power:
            high = mid
        else:
            low = mid
    _ = df
    return high


def required_pairs(
    *, effect: float, sigma_d: float, alpha: float = 0.05, power: float = DEFAULT_POWER
) -> int:
    """Number of paired tasks needed to detect ``effect`` at the target power.

    Args:
        effect: Effect to detect (the gate's margin, typically).
        sigma_d: SD of the per-task differences.
        alpha: Significance level.
        power: Target power.

    Returns:
        The required ``n``, or :data:`MAX_N` when the effect is undetectable at any practical
        sample size — which is itself the finding, and the report says so.
    """
    if effect <= 0.0:
        return MAX_N
    if sigma_d <= 0.0:
        return 2
    low, high = 2, 64
    while (
        high < MAX_N and paired_power(effect=effect, sigma_d=sigma_d, n=high, alpha=alpha) < power
    ):
        low = high
        high *= 2
    if high >= MAX_N:
        return MAX_N
    while low < high:
        mid = (low + high) // 2
        if paired_power(effect=effect, sigma_d=sigma_d, n=mid, alpha=alpha) >= power:
            high = mid
        else:
            low = mid + 1
    return low


def mcnemar_power(
    *, discordant_rate: float, odds_ratio: float, n: int, alpha: float = 0.05
) -> float:
    """Approximate power of a one-sided McNemar test.

    Uses the standard normal approximation to the conditional binomial: with ``n * p_d``
    expected discordant pairs and a true split ``psi = OR/(1+OR)``, power is the probability
    that the observed split is extreme enough to reject a fair coin.

    Assumptions, stated because they matter at the sample sizes this project targets: the
    normal approximation to the binomial degrades below roughly 10 expected discordant pairs, so
    the number is reported as approximate and the gate prefers the paired-t path on task-level
    proportions when K > 1.

    Args:
        discordant_rate: Expected share of tasks on which the two systems disagree.
        odds_ratio: True ratio of candidate-only to baseline-only successes.
        n: Number of paired tasks.
        alpha: Significance level.

    Returns:
        Approximate power in [0, 1].
    """
    expected_discordant = n * discordant_rate
    if expected_discordant <= 0 or odds_ratio <= 0:
        return 0.0
    psi = odds_ratio / (1.0 + odds_ratio)
    z_alpha = float(stats.norm.ppf(1.0 - alpha))
    numerator = abs(psi - 0.5) * math.sqrt(expected_discordant) - z_alpha * 0.5
    denominator = math.sqrt(psi * (1.0 - psi))
    if denominator == 0.0:
        return 1.0
    return float(min(1.0, max(0.0, stats.norm.cdf(numerator / denominator))))


@dataclass(frozen=True, slots=True)
class PowerInputs:
    """What a power calculation needs."""

    sigma_d: float
    n_pairs: int
    margin: float
    alpha: float = 0.05
    target: float = DEFAULT_POWER


def power_report(inputs: PowerInputs, *, method: str = "paired-t (non-central)") -> PowerReport:
    """Build the power panel for one gated metric.

    Args:
        inputs: Dispersion, sample size, margin, and levels.
        method: Label recorded on the report.

    Returns:
        Achieved power for the configured margin, the MDE at this ``n``, and the ``n`` that
        would be needed.
    """
    achieved = paired_power(
        effect=inputs.margin, sigma_d=inputs.sigma_d, n=inputs.n_pairs, alpha=inputs.alpha
    )
    mde = minimum_detectable_effect(
        sigma_d=inputs.sigma_d, n=inputs.n_pairs, alpha=inputs.alpha, power=inputs.target
    )
    needed = required_pairs(
        effect=inputs.margin, sigma_d=inputs.sigma_d, alpha=inputs.alpha, power=inputs.target
    )
    return PowerReport(
        sigma_d=max(0.0, inputs.sigma_d),
        n_pairs=max(0, inputs.n_pairs),
        alpha=inputs.alpha,
        power_target=inputs.target,
        achieved_power=achieved,
        mde=0.0 if not math.isfinite(mde) else mde,
        n_required=needed,
        method=method,
    )


def describe(report: PowerReport, metric: str, margin: float) -> str:
    """The sentence every gate report prints (C4)."""
    if report.n_pairs < 2:
        return f"{metric}: too few paired tasks to say anything about power."
    return (
        f"This suite can detect a change of {report.mde:.3f} in {metric} with "
        f"{report.power_target:.0%} power at n={report.n_pairs}; your configured margin is "
        f"delta={margin:.3f}, for which achieved power is {report.achieved_power:.0%} "
        f"(n={report.n_required} pairs would be needed for {report.power_target:.0%})."
    )


def plan_suite_size(
    *, target_mde: float, sigma_d: float, alpha: float = 0.05, power: float = DEFAULT_POWER
) -> dict[str, float]:
    """Answer "how many tasks should I write?" — the ``agentgate plan`` calculation.

    Args:
        target_mde: The smallest regression the author wants to be able to detect.
        sigma_d: SD of per-task differences, estimated from history or a pilot run.
        alpha: Significance level.
        power: Target power.

    Returns:
        A summary with the required task count and the assumptions behind it.
    """
    needed = required_pairs(effect=target_mde, sigma_d=sigma_d, alpha=alpha, power=power)
    return {
        "target_mde": target_mde,
        "sigma_d": sigma_d,
        "alpha": alpha,
        "power": power,
        "required_tasks": float(needed),
        "achieved_power_at_required": paired_power(
            effect=target_mde, sigma_d=sigma_d, n=needed, alpha=alpha
        ),
    }
