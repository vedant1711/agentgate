"""Paired comparison — the gate's engine (C2).

Pairing on identical task instances and seeds is the single highest-leverage decision in the
whole design. Two versions of an agent correlate 0.3-0.7 on per-question scores (E3), and a
paired analysis converts that correlation directly into variance reduction:

``Var(d) = Var(x) + Var(y) - 2*r*SD(x)*SD(y)``

At r = 0.5 with equal variances, the paired variance is *half* the unpaired variance — the same
precision for half the sample, for free. The report prints the realised ``r`` so the reader can
see what pairing bought on their suite rather than taking the claim on faith.

Test selection follows the metric's dtype and the data's shape, and the choice is always logged:
a reader must be able to tell whether a p-value came from a t-test or a Wilcoxon.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from agentgate.errors import InsufficientDataError
from agentgate.schemas.results import EffectSize, Estimate, PairedTestResult
from agentgate.stats.intervals import z_for

Alternative = Literal["less", "greater", "two-sided"]
MIN_PAIRS = 2
DEFAULT_PERMUTATIONS = 20_000


@dataclass(frozen=True, slots=True)
class PairedData:
    """Aligned per-task scores for two systems."""

    task_ids: tuple[str, ...]
    baseline: tuple[float, ...]
    candidate: tuple[float, ...]
    clusters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate alignment."""
        if not (len(self.task_ids) == len(self.baseline) == len(self.candidate)):
            msg = "paired data must be aligned across task ids, baseline, and candidate"
            raise InsufficientDataError(msg)

    @property
    def n(self) -> int:
        """Number of paired tasks."""
        return len(self.task_ids)

    @property
    def differences(self) -> list[float]:
        """Per-task ``candidate - baseline`` differences."""
        return [c - b for b, c in zip(self.baseline, self.candidate, strict=True)]

    @property
    def has_clusters(self) -> bool:
        """True when cluster ids were supplied and are not all distinct."""
        return bool(self.clusters) and len(set(self.clusters)) < self.n

    def directed(self, direction: str) -> PairedData:
        """Return data direction-normalised so that higher is always better.

        For ``lower_is_better`` metrics both series are negated, which turns "latency went up"
        into "the score went down" and lets one set of one-sided tests serve every metric.
        """
        if direction == "higher_is_better":
            return self
        return PairedData(
            task_ids=self.task_ids,
            baseline=tuple(-value for value in self.baseline),
            candidate=tuple(-value for value in self.candidate),
            clusters=self.clusters,
        )


def paired_correlation(data: PairedData) -> float | None:
    """Pearson correlation between the paired scores — the variance pairing bought (E3)."""
    if data.n < 3:
        return None
    baseline = np.asarray(data.baseline, dtype=float)
    candidate = np.asarray(data.candidate, dtype=float)
    if baseline.std() == 0.0 or candidate.std() == 0.0:
        return None
    return float(np.corrcoef(baseline, candidate)[0, 1])


def difference_estimate(
    data: PairedData, *, level: float = 0.95, use_clusters: bool = True
) -> Estimate:
    """Mean paired difference with its interval.

    Uses the cluster-robust SE when clusters exist, computed **directly on the per-cluster mean
    differences** (E3's recommendation) rather than on the raw differences.

    Args:
        data: Aligned paired scores.
        level: Confidence level.
        use_clusters: Whether to use the clustered SE when clusters are present.

    Returns:
        The mean difference estimate.
    """
    differences = data.differences
    n = len(differences)
    if n == 0:
        return Estimate(value=float("nan"), method="paired(empty)", n=0)
    mean = float(np.mean(differences))
    if n < MIN_PAIRS:
        return Estimate(value=mean, method="paired(n<2)", n=n)

    if use_clusters and data.has_clusters:
        groups: dict[str, list[float]] = {}
        for value, cluster in zip(differences, data.clusters, strict=True):
            groups.setdefault(cluster, []).append(value)
        cluster_means = [float(np.mean(values)) for values in groups.values()]
        g = len(cluster_means)
        if g >= MIN_PAIRS:
            se = float(np.std(cluster_means, ddof=1) / math.sqrt(g))
            half = z_for(level) * se
            return Estimate(
                value=float(np.mean(cluster_means)),
                se=se,
                ci_low=float(np.mean(cluster_means)) - half,
                ci_high=float(np.mean(cluster_means)) + half,
                ci_level=level,
                method="paired-clustered",
                n=g,
            )

    se = float(np.std(differences, ddof=1) / math.sqrt(n))
    half = z_for(level) * se
    return Estimate(
        value=mean,
        se=se,
        ci_low=mean - half,
        ci_high=mean + half,
        ci_level=level,
        method="paired-clt",
        n=n,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def paired_t_test(
    differences: Sequence[float], *, shift: float = 0.0, alternative: Alternative = "two-sided"
) -> PairedTestResult:
    """One-sample t-test on shifted paired differences.

    The shift implements the non-inferiority margin: testing ``d_i + delta`` against zero is
    testing ``mean(d) > -delta``, which is the hypothesis the gate actually cares about (C3).

    Args:
        differences: Per-task ``candidate - baseline``.
        shift: Added to every difference before testing (the margin ``delta``).
        alternative: ``greater`` rejects "the mean is at most zero"; ``less`` rejects "at least".

    Returns:
        The test result.

    Raises:
        InsufficientDataError: With fewer than two pairs.
    """
    shifted = np.asarray(differences, dtype=float) + shift
    n = int(shifted.size)
    if n < MIN_PAIRS:
        msg = f"paired t-test needs at least {MIN_PAIRS} pairs, got {n}"
        raise InsufficientDataError(msg)
    sd = float(shifted.std(ddof=1))
    mean = float(shifted.mean())
    if sd == 0.0:
        # Zero-variance differences: the t statistic is +/-inf or undefined. Report the honest
        # degenerate p-value rather than dividing by zero.
        p_one = (
            0.0
            if (mean > 0 and alternative == "greater") or (mean < 0 and alternative == "less")
            else 1.0
        )
        return PairedTestResult(
            test="paired_t",
            statistic=None,
            p_one_sided=p_one,
            p_two_sided=0.0 if mean != 0.0 else 1.0,
            df=float(n - 1),
            n_pairs=n,
            detail={"mean_shifted": mean, "sd": 0.0},
            selection_reason="degenerate: all shifted differences identical",
        )
    t_stat = mean / (sd / math.sqrt(n))
    df = n - 1
    if alternative == "greater":
        p_one = float(stats.t.sf(t_stat, df))
    elif alternative == "less":
        p_one = float(stats.t.cdf(t_stat, df))
    else:
        p_one = float(2.0 * stats.t.sf(abs(t_stat), df))
    return PairedTestResult(
        test="paired_t",
        statistic=float(t_stat),
        p_one_sided=min(1.0, max(0.0, p_one)),
        p_two_sided=min(1.0, float(2.0 * stats.t.sf(abs(t_stat), df))),
        df=float(df),
        n_pairs=n,
        detail={"mean_shifted": mean, "sd": sd},
    )


def wilcoxon_test(
    differences: Sequence[float], *, shift: float = 0.0, alternative: Alternative = "two-sided"
) -> PairedTestResult:
    """Wilcoxon signed-rank test on shifted paired differences.

    The non-parametric fallback when the differences are not plausibly normal. It tests a shift
    in the *distribution* rather than the mean, which is a slightly different claim — noted in
    the result so a reader is not misled about what was tested.
    """
    shifted = np.asarray(differences, dtype=float) + shift
    n = int(shifted.size)
    if n < MIN_PAIRS:
        msg = f"Wilcoxon needs at least {MIN_PAIRS} pairs, got {n}"
        raise InsufficientDataError(msg)
    if np.all(shifted == 0.0):
        return PairedTestResult(
            test="wilcoxon",
            statistic=0.0,
            p_one_sided=1.0,
            p_two_sided=1.0,
            n_pairs=n,
            selection_reason="degenerate: every shifted difference is zero",
        )
    outcome = stats.wilcoxon(shifted, alternative=alternative, zero_method="zsplit")
    two_sided = stats.wilcoxon(shifted, alternative="two-sided", zero_method="zsplit")
    return PairedTestResult(
        test="wilcoxon",
        statistic=float(outcome.statistic),
        p_one_sided=float(outcome.pvalue),
        p_two_sided=float(two_sided.pvalue),
        n_pairs=n,
        detail={"median_shifted": float(np.median(shifted))},
        selection_reason="tests a distributional shift, not a mean shift",
    )


def mcnemar_exact(discordant_baseline: int, discordant_candidate: int) -> PairedTestResult:
    """Exact McNemar test on the discordant-pair table (McNemar, 1947).

    Concordant pairs carry no information about a *difference*: a task both systems passed
    tells you nothing about which is better. So the test conditions on the ``b + c`` discordant
    pairs and asks whether the split is consistent with a fair coin.

    Args:
        discordant_baseline: ``b`` — tasks the baseline passed and the candidate failed.
        discordant_candidate: ``c`` — tasks the candidate passed and the baseline failed.

    Returns:
        The test result. ``p_one_sided`` tests "the candidate is worse" — the direction the gate
        cares about. The odds ratio ``c/b`` is reported in the detail.
    """
    b, c = int(discordant_baseline), int(discordant_candidate)
    total = b + c
    if total == 0:
        return PairedTestResult(
            test="mcnemar_exact",
            statistic=0.0,
            p_one_sided=1.0,
            p_two_sided=1.0,
            n_pairs=0,
            detail={"b": 0.0, "c": 0.0},
            selection_reason="no discordant pairs: the systems never disagreed",
        )
    p_one = float(stats.binom.cdf(c, total, 0.5))
    p_two = float(min(1.0, 2.0 * stats.binom.cdf(min(b, c), total, 0.5)))
    odds_ratio = (c / b) if b else float("inf")
    return PairedTestResult(
        test="mcnemar_exact",
        statistic=float(c),
        p_one_sided=min(1.0, max(0.0, p_one)),
        p_two_sided=p_two,
        n_pairs=total,
        detail={"b": float(b), "c": float(c), "odds_ratio": odds_ratio},
        selection_reason="exact binomial on discordant pairs",
    )


def discordant_counts(data: PairedData, *, threshold: float = 0.5) -> tuple[int, int]:
    """Count discordant pairs from paired binary (or thresholded) scores.

    Args:
        data: Paired scores.
        threshold: Scores at or above this count as a success — needed because a task's score is
            a *mean over K repetitions*, not a single bit.

    Returns:
        ``(b, c)``: baseline-only successes, candidate-only successes.
    """
    b = c = 0
    for base, cand in zip(data.baseline, data.candidate, strict=True):
        base_ok = base >= threshold
        cand_ok = cand >= threshold
        if base_ok and not cand_ok:
            b += 1
        elif cand_ok and not base_ok:
            c += 1
    return b, c


def permutation_test(
    differences: Sequence[float],
    *,
    shift: float = 0.0,
    alternative: Alternative = "two-sided",
    iterations: int = DEFAULT_PERMUTATIONS,
    seed: int = 20260101,
) -> PairedTestResult:
    """Sign-flip permutation test on paired differences.

    Under the null of exchangeability, the sign of each paired difference is arbitrary. Flipping
    signs at random builds the null distribution directly from the data, assuming nothing about
    its shape — which is why the report runs it alongside the parametric test and flags material
    disagreement rather than picking a winner in advance.

    The p-value uses the ``(1 + hits) / (1 + B)`` convention, which keeps it strictly positive:
    a p-value of exactly 0 would claim more evidence than B resamples can provide.
    """
    shifted = np.asarray(differences, dtype=float) + shift
    n = int(shifted.size)
    if n < MIN_PAIRS:
        msg = f"permutation test needs at least {MIN_PAIRS} pairs, got {n}"
        raise InsufficientDataError(msg)

    observed = float(shifted.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(iterations, n))
    null_means = (signs * shifted).mean(axis=1)

    if alternative == "greater":
        hits = int(np.sum(null_means >= observed))
    elif alternative == "less":
        hits = int(np.sum(null_means <= observed))
    else:
        hits = int(np.sum(np.abs(null_means) >= abs(observed)))
    p_value = (1.0 + hits) / (1.0 + iterations)
    two_sided = (1.0 + int(np.sum(np.abs(null_means) >= abs(observed)))) / (1.0 + iterations)
    return PairedTestResult(
        test="permutation",
        statistic=observed,
        p_one_sided=p_value,
        p_two_sided=two_sided,
        n_pairs=n,
        detail={"iterations": float(iterations)},
        selection_reason="sign-flip null; assumes exchangeability only",
    )


# ---------------------------------------------------------------------------
# Test selection and effect sizes
# ---------------------------------------------------------------------------


def choose_test(differences: Sequence[float], *, normality_alpha: float = 0.05) -> tuple[str, str]:
    """Pick the parametric or non-parametric paired test, and say why.

    Shapiro-Wilk on the differences at ``normality_alpha``; a rejection sends the analysis to
    Wilcoxon. The reason string is recorded in the result so no p-value in a report is of
    unknown provenance.

    Args:
        differences: Per-task differences.
        normality_alpha: Rejection level for the normality check.

    Returns:
        ``(test name, reason)``.
    """
    values = np.asarray(differences, dtype=float)
    n = int(values.size)
    if n < 3:
        return "paired_t", f"n={n} is too small to test normality; defaulting to paired t"
    if float(values.std(ddof=1)) == 0.0:
        return "paired_t", "all differences identical; normality test is undefined"
    if n > 5000:
        return "paired_t", "n>5000; CLT makes the t-test robust regardless of shape"
    shapiro_p = float(stats.shapiro(values).pvalue)
    if shapiro_p < normality_alpha:
        return "wilcoxon", f"Shapiro-Wilk p={shapiro_p:.4f} < {normality_alpha}: non-normal"
    return "paired_t", f"Shapiro-Wilk p={shapiro_p:.4f} >= {normality_alpha}: normal enough"


def run_paired_test(
    differences: Sequence[float],
    *,
    shift: float = 0.0,
    alternative: Alternative = "two-sided",
    normality_alpha: float = 0.05,
    forced: str | None = None,
    iterations: int = DEFAULT_PERMUTATIONS,
    seed: int = 20260101,
) -> PairedTestResult:
    """Run whichever paired test the data warrants, recording the choice.

    Args:
        differences: Per-task (or per-cluster) differences.
        shift: Non-inferiority margin added before testing.
        alternative: Test direction.
        normality_alpha: Shapiro-Wilk threshold.
        forced: Force ``"paired_t"``, ``"wilcoxon"``, or ``"permutation"``.
        iterations: Resamples, when the permutation test is used.
        seed: RNG seed for the permutation test.

    Returns:
        The test result, with ``selection_reason`` explaining the choice.
    """
    name, reason = (
        (forced, f"forced: {forced}")
        if forced
        else choose_test(differences, normality_alpha=normality_alpha)
    )
    if name == "wilcoxon" and shift != 0.0 and not forced:
        # A non-inferiority margin is defined on the **mean**, and Wilcoxon tests a median or
        # distributional shift. On skewed differences the two genuinely disagree — a candidate
        # whose mean sits comfortably inside the margin can still have a median outside it — and
        # answering the wrong question with a small p-value is worse than answering nothing.
        # The permutation test is distribution-free *and* about the mean, so it is used instead.
        name = "permutation"
        reason = (
            f"{reason}; switched to the sign-flip permutation test because a non-inferiority "
            f"margin is defined on the mean while Wilcoxon tests a median shift"
        )

    if name == "wilcoxon":
        result = wilcoxon_test(differences, shift=shift, alternative=alternative)
    elif name == "permutation":
        result = permutation_test(
            differences,
            shift=shift,
            alternative=alternative,
            iterations=iterations,
            seed=seed,
        )
    else:
        result = paired_t_test(differences, shift=shift, alternative=alternative)
    return result.model_copy(update={"selection_reason": reason})


def cohens_dz(differences: Sequence[float]) -> EffectSize | None:
    """Cohen's d_z for a paired design: mean difference over the SD of the differences.

    Note the ``_z``: this standardises by the SD of the *differences*, not of the raw scores.
    The two coincide only when the paired correlation is zero, and reporting one as the other
    inflates the apparent effect on well-paired data.
    """
    values = np.asarray(differences, dtype=float)
    if values.size < MIN_PAIRS:
        return None
    sd = float(values.std(ddof=1))
    if sd == 0.0:
        return None
    value = float(values.mean() / sd)
    magnitude = abs(value)
    label = (
        "negligible"
        if magnitude < 0.2
        else "small"
        if magnitude < 0.5
        else "medium"
        if magnitude < 0.8
        else "large"
    )
    return EffectSize(kind="cohens_dz", value=value, interpretation=label)


def mcnemar_odds_ratio(b: int, c: int) -> EffectSize | None:
    """Odds ratio ``c/b`` from the discordant-pair table."""
    if b == 0 and c == 0:
        return None
    if b == 0:
        return EffectSize(kind="odds_ratio", value=float("inf"), interpretation="candidate only")
    value = c / b
    label = "candidate better" if value > 1 else "baseline better" if value < 1 else "even"
    return EffectSize(kind="odds_ratio", value=value, interpretation=label)


DECISIVE_P = 0.005
"""Below this both tests agree decisively; their ratio cannot change any verdict."""

INDECISIVE_P = 0.5
"""Above this both tests are decisively non-significant, and the ratio is equally moot."""


def disagreement(parametric: PairedTestResult, permutation: PairedTestResult) -> str | None:
    """Flag *material* disagreement between the parametric and permutation p-values.

    Material means the two could support different decisions:

    * they land on opposite sides of 0.05, or
    * they differ by more than 2x while both sit in the ambiguous band where a different alpha
      would change the answer.

    Deliberately silent when both are decisive. A permutation p-value is floored at
    ``1/(B+1)``, so comparing it by *ratio* against a parametric p of 1e-8 always looks like a
    disagreement and never is one — flagging that would train readers to ignore the flag.
    """
    p_param = parametric.p_one_sided
    p_perm = permutation.p_one_sided
    if (p_param < 0.05) != (p_perm < 0.05):
        return (
            f"parametric ({parametric.test}) p={p_param:.4f} and permutation p={p_perm:.4f} "
            f"fall on opposite sides of 0.05; the parametric assumptions are doing real work here"
        )
    lower, upper = min(p_param, p_perm), max(p_param, p_perm)
    ambiguous = lower >= DECISIVE_P and upper <= INDECISIVE_P
    if ambiguous and upper / max(1e-12, lower) > 2.0:
        return (
            f"parametric ({parametric.test}) p={p_param:.4f} and permutation p={p_perm:.4f} "
            f"differ by more than 2x in the range where alpha choice matters"
        )
    return None
