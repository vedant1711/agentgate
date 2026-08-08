"""Statistics engine unit tests — closed-form checks against textbook values.

The Monte-Carlo calibration (type-I error, coverage, power, unbiasedness) lives in
``tests/simulation/``. These tests pin the arithmetic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as scipy_stats

from agentgate.errors import InsufficientDataError
from agentgate.stats import (
    PairedData,
    PowerInputs,
    benjamini_hochberg,
    bonferroni,
    bootstrap_bca,
    choose_test,
    clt_interval,
    clustered_interval,
    cohens_dz,
    decompose,
    describe,
    difference_estimate,
    disagreement,
    discordant_counts,
    expected_false_alarms,
    family_wise_error,
    flake_rate,
    mcnemar_exact,
    minimum_detectable_effect,
    naive_pass_hat_k,
    outcomes_from_scores,
    paired_correlation,
    paired_power,
    paired_t_test,
    pass_at_k,
    pass_hat_k,
    percentile_interval,
    permutation_test,
    power_report,
    propagate_judge_variance,
    reliability_report,
    required_pairs,
    run_paired_test,
    se_inflation,
    wilcoxon_test,
    wilson_interval,
    z_for,
)

# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


def test_z_for_matches_the_familiar_critical_values() -> None:
    assert z_for(0.95) == pytest.approx(1.959964, abs=1e-5)
    assert z_for(0.99) == pytest.approx(2.575829, abs=1e-5)


def test_clt_interval_matches_hand_computation() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    estimate = clt_interval(values)
    # mean 3, sd 1.5811388, se = 1.5811388/sqrt(5) = 0.7071068
    assert estimate.value == pytest.approx(3.0)
    assert estimate.se == pytest.approx(0.7071068, abs=1e-6)
    assert estimate.ci_low == pytest.approx(3.0 - 1.959964 * 0.7071068, abs=1e-5)
    assert estimate.n == 5


def test_clt_interval_reports_no_interval_below_two_observations() -> None:
    single = clt_interval([0.5])
    assert single.value == 0.5
    assert single.se is None
    assert "n<2" in single.method
    assert clt_interval([]).n == 0


def test_wilson_interval_matches_a_published_example() -> None:
    """p-hat = 0.8 at n = 10; Wilson 95% interval is roughly (0.4902, 0.9433)."""
    estimate = wilson_interval(8, 10)
    assert estimate.value == pytest.approx(0.8)
    assert estimate.ci_low == pytest.approx(0.4902, abs=1e-3)
    assert estimate.ci_high == pytest.approx(0.9433, abs=1e-3)


def test_wilson_stays_inside_the_unit_interval_at_the_boundary() -> None:
    """The normal approximation would extend past 1.0 here; Wilson does not."""
    perfect = wilson_interval(10, 10)
    assert perfect.value == 1.0
    assert perfect.ci_high == pytest.approx(1.0)
    assert perfect.ci_low is not None
    assert 0.0 < perfect.ci_low < 1.0

    zero = wilson_interval(0, 10)
    assert zero.ci_low == pytest.approx(0.0)
    assert zero.ci_high is not None
    assert 0.0 < zero.ci_high < 1.0


def test_wilson_interval_always_contains_its_point_estimate() -> None:
    """Floating point puts the p=1 bound an ULP short; an interval must contain its estimate."""
    for successes, trials in ((13, 13), (0, 13), (1, 3), (7, 9)):
        estimate = wilson_interval(successes, trials)
        assert estimate.ci_low is not None and estimate.ci_high is not None
        assert estimate.ci_low <= estimate.value <= estimate.ci_high


def test_wilson_needs_a_trial() -> None:
    with pytest.raises(InsufficientDataError, match="at least one trial"):
        wilson_interval(0, 0)


def test_clustered_se_exceeds_naive_when_clusters_are_homogeneous() -> None:
    """The >3x phenomenon E3 warns about, reproduced deterministically."""
    values = [0.9] * 10 + [0.1] * 10
    clusters = ["a"] * 10 + ["b"] * 10
    naive = clt_interval(values)
    clustered = clustered_interval(values, clusters)

    assert clustered.value == pytest.approx(naive.value)
    assert clustered.se is not None and naive.se is not None
    assert clustered.se > naive.se
    assert clustered.n == 2, "the honest denominator is the number of clusters"
    ratio = se_inflation(naive, clustered)
    assert ratio is not None and ratio > 2.0


def test_clustered_se_matches_naive_when_every_task_is_its_own_cluster() -> None:
    values = [0.1, 0.5, 0.9, 0.3, 0.7]
    clusters = [f"c{i}" for i in range(5)]
    naive = clt_interval(values)
    clustered = clustered_interval(values, clusters)
    assert clustered.se == pytest.approx(naive.se, rel=0.01)


def test_clustered_interval_refuses_misaligned_inputs() -> None:
    with pytest.raises(InsufficientDataError, match="aligned"):
        clustered_interval([1.0, 2.0], ["a"])


def test_bootstrap_bca_brackets_its_own_estimate() -> None:
    """The interval is well-formed and its width matches the analytic SE for a mean.

    Coverage of the *population* mean is a Monte-Carlo question, checked in H3.
    """
    rng = np.random.default_rng(7)
    values = rng.normal(0.6, 0.2, size=200).tolist()
    estimate = bootstrap_bca(values, lambda s: float(np.mean(s)), resamples=2000, seed=1)
    sample_mean = float(np.mean(values))
    assert estimate.value == pytest.approx(sample_mean)
    assert estimate.ci_low is not None and estimate.ci_high is not None
    assert estimate.ci_low < sample_mean < estimate.ci_high
    analytic_se = float(np.std(values, ddof=1)) / math.sqrt(len(values))
    assert estimate.se == pytest.approx(analytic_se, rel=0.15)


def test_bootstrap_bca_is_reproducible() -> None:
    values = [0.1, 0.4, 0.5, 0.9, 0.3, 0.7, 0.2, 0.8]
    first = bootstrap_bca(values, lambda s: float(np.mean(s)), resamples=500, seed=3)
    second = bootstrap_bca(values, lambda s: float(np.mean(s)), resamples=500, seed=3)
    assert first.ci_low == second.ci_low
    assert first.ci_high == second.ci_high


def test_bootstrap_resamples_clusters_not_tasks() -> None:
    """Cluster resampling must widen the interval relative to naive resampling."""
    values = [0.9] * 12 + [0.1] * 12
    clusters = ["a"] * 12 + ["b"] * 12
    naive = bootstrap_bca(values, lambda s: float(np.mean(s)), resamples=800, seed=5)
    grouped = bootstrap_bca(
        values, lambda s: float(np.mean(s)), clusters=clusters, resamples=800, seed=5
    )
    assert grouped.se is not None and naive.se is not None
    assert grouped.se > naive.se


def test_bootstrap_handles_a_degenerate_sample() -> None:
    estimate = bootstrap_bca([0.5] * 10, lambda s: float(np.mean(s)), resamples=200, seed=1)
    assert estimate.value == pytest.approx(0.5)
    assert "percentile" in estimate.method or estimate.ci_low == pytest.approx(0.5)


def test_percentile_interval_is_symmetric_on_symmetric_data() -> None:
    low, high = percentile_interval(list(np.linspace(0.0, 1.0, 1001)))
    assert low == pytest.approx(0.025, abs=1e-3)
    assert high == pytest.approx(0.975, abs=1e-3)


# ---------------------------------------------------------------------------
# Paired tests
# ---------------------------------------------------------------------------


def make_paired(
    baseline: list[float], candidate: list[float], clusters: list[str] | None = None
) -> PairedData:
    return PairedData(
        task_ids=tuple(f"t{i}" for i in range(len(baseline))),
        baseline=tuple(baseline),
        candidate=tuple(candidate),
        clusters=tuple(clusters or [f"c{i}" for i in range(len(baseline))]),
    )


def test_paired_t_matches_scipy() -> None:
    differences = [0.1, -0.2, 0.05, 0.3, -0.1, 0.2, 0.0, 0.15]
    ours = paired_t_test(differences, alternative="greater")
    theirs = scipy_stats.ttest_1samp(differences, 0.0, alternative="greater")
    assert ours.statistic == pytest.approx(float(theirs.statistic))
    assert ours.p_one_sided == pytest.approx(float(theirs.pvalue))


def test_shifting_by_the_margin_tests_the_right_hypothesis() -> None:
    """A candidate 2 points worse is non-inferior at a 5-point margin, inferior at 1 point."""
    differences = [-0.02] * 30
    lenient = paired_t_test(differences, shift=0.05, alternative="greater")
    strict = paired_t_test(differences, shift=0.01, alternative="less")
    assert lenient.p_one_sided < 0.01, "non-inferiority is established at delta=0.05"
    assert strict.p_one_sided < 0.01, "regression beyond delta=0.01 is established"


def test_paired_t_handles_zero_variance_without_dividing_by_zero() -> None:
    result = paired_t_test([0.1] * 10, alternative="greater")
    assert result.p_one_sided == 0.0
    assert result.statistic is None
    assert "degenerate" in result.selection_reason


def test_paired_t_needs_two_pairs() -> None:
    with pytest.raises(InsufficientDataError, match="at least 2 pairs"):
        paired_t_test([0.1])


def test_wilcoxon_matches_scipy() -> None:
    differences = [0.3, 0.1, -0.05, 0.4, 0.25, -0.02, 0.35, 0.15]
    ours = wilcoxon_test(differences, alternative="greater")
    theirs = scipy_stats.wilcoxon(differences, alternative="greater", zero_method="zsplit")
    assert ours.p_one_sided == pytest.approx(float(theirs.pvalue))


def test_wilcoxon_on_all_zero_differences_is_not_significant() -> None:
    result = wilcoxon_test([0.0] * 8)
    assert result.p_one_sided == 1.0
    assert "degenerate" in result.selection_reason


def test_mcnemar_exact_matches_the_binomial() -> None:
    """b=10, c=2: one-sided p = P(X <= 2) for X ~ Binom(12, 0.5) = 0.019287."""
    result = mcnemar_exact(10, 2)
    assert result.p_one_sided == pytest.approx(0.0192871, abs=1e-6)
    assert result.p_two_sided == pytest.approx(0.0385742, abs=1e-6)
    assert result.detail["odds_ratio"] == pytest.approx(0.2)


def test_mcnemar_ignores_concordant_pairs() -> None:
    """A task both systems passed says nothing about which is better."""
    assert mcnemar_exact(3, 3).p_two_sided == pytest.approx(1.0)
    assert mcnemar_exact(0, 0).p_one_sided == 1.0
    assert mcnemar_exact(0, 0).n_pairs == 0


def test_mcnemar_with_no_baseline_only_successes_is_infinite_odds() -> None:
    assert mcnemar_exact(0, 5).detail["odds_ratio"] == float("inf")


def test_discordant_counts_threshold_per_task_rate() -> None:
    data = make_paired([1.0, 1.0, 0.0, 0.25], [0.0, 1.0, 1.0, 0.75])
    b, c = discordant_counts(data, threshold=0.5)
    assert (b, c) == (1, 2)


def test_permutation_p_is_bounded_away_from_zero() -> None:
    result = permutation_test([0.5] * 20, alternative="greater", iterations=1000, seed=1)
    assert result.p_one_sided >= 1.0 / 1001
    assert result.p_one_sided <= 1.0


def test_permutation_agrees_with_the_t_test_on_normal_data() -> None:
    rng = np.random.default_rng(11)
    differences = rng.normal(0.15, 0.2, size=60).tolist()
    parametric = paired_t_test(differences, alternative="greater")
    resampled = permutation_test(differences, alternative="greater", iterations=20000, seed=2)
    assert resampled.p_one_sided == pytest.approx(parametric.p_one_sided, abs=0.02)
    assert disagreement(parametric, resampled) is None, (
        "both p-values are decisively below alpha; their ratio at the permutation floor is "
        "not a disagreement"
    )


def test_disagreement_is_flagged_when_the_tests_straddle_alpha() -> None:
    from agentgate.schemas.results import PairedTestResult

    parametric = PairedTestResult(test="paired_t", p_one_sided=0.02, n_pairs=10)
    resampled = PairedTestResult(test="permutation", p_one_sided=0.20, n_pairs=10)
    flagged = disagreement(parametric, resampled)
    assert flagged is not None
    assert "opposite sides of 0.05" in flagged


def test_disagreement_is_silent_when_both_tests_are_decisive() -> None:
    """A ratio at the permutation resolution floor is arithmetic, not a finding."""
    from agentgate.schemas.results import PairedTestResult

    parametric = PairedTestResult(test="paired_t", p_one_sided=1e-8, n_pairs=60)
    resampled = PairedTestResult(test="permutation", p_one_sided=5e-5, n_pairs=60)
    assert disagreement(parametric, resampled) is None


def test_disagreement_is_flagged_inside_the_ambiguous_band() -> None:
    from agentgate.schemas.results import PairedTestResult

    parametric = PairedTestResult(test="paired_t", p_one_sided=0.08, n_pairs=40)
    resampled = PairedTestResult(test="permutation", p_one_sided=0.30, n_pairs=40)
    flagged = disagreement(parametric, resampled)
    assert flagged is not None
    assert "alpha choice matters" in flagged


def test_test_selection_prefers_wilcoxon_on_skewed_differences() -> None:
    rng = np.random.default_rng(3)
    skewed = (rng.exponential(0.3, size=60) - 0.1).tolist()
    name, reason = choose_test(skewed)
    assert name == "wilcoxon"
    assert "Shapiro-Wilk" in reason


def test_test_selection_prefers_t_on_normal_differences() -> None:
    rng = np.random.default_rng(4)
    name, reason = choose_test(rng.normal(0.0, 0.1, size=80).tolist())
    assert name == "paired_t"
    assert "normal enough" in reason


def test_selection_reason_is_always_recorded() -> None:
    result = run_paired_test([0.1, 0.2, -0.05, 0.3, 0.1, 0.0, 0.25, 0.15])
    assert result.selection_reason


def test_forced_test_is_honoured_and_labelled() -> None:
    result = run_paired_test([0.1, 0.2, 0.3, 0.4], forced="wilcoxon")
    assert result.test == "wilcoxon"
    assert "forced" in result.selection_reason


def test_cohens_dz_standardises_by_the_difference_sd() -> None:
    differences = [0.2, 0.2, 0.2, 0.2, 0.4, 0.0]
    effect = cohens_dz(differences)
    assert effect is not None
    assert effect.value == pytest.approx(
        float(np.mean(differences)) / float(np.std(differences, ddof=1))
    )
    assert effect.kind == "cohens_dz"


def test_cohens_dz_is_none_without_variance() -> None:
    assert cohens_dz([0.3] * 5) is None


def test_pairing_correlation_is_reported() -> None:
    data = make_paired([0.1, 0.4, 0.7, 0.9, 0.5], [0.15, 0.45, 0.72, 0.88, 0.52])
    correlation = paired_correlation(data)
    assert correlation is not None
    assert correlation > 0.95


def test_pairing_halves_the_variance_at_r_equals_one_half() -> None:
    """The E3 claim, checked arithmetically rather than asserted."""
    rng = np.random.default_rng(19)
    n = 4000
    shared = rng.normal(0, 1, n)
    baseline = shared + rng.normal(0, 1, n)
    candidate = shared + rng.normal(0, 1, n)
    paired_variance = float(np.var(candidate - baseline, ddof=1))
    unpaired_variance = float(np.var(baseline, ddof=1) + np.var(candidate, ddof=1))
    assert paired_variance == pytest.approx(unpaired_variance * 0.5, rel=0.1)


def test_direction_normalisation_flips_lower_is_better_metrics() -> None:
    latency = make_paired([100.0, 120.0], [90.0, 110.0])
    normalised = latency.directed("lower_is_better")
    assert normalised.differences == [10.0, 10.0], "getting faster is an improvement"
    assert latency.directed("higher_is_better").differences == [-10.0, -10.0]


def test_difference_estimate_uses_cluster_mean_differences() -> None:
    data = make_paired([0.9] * 6 + [0.1] * 6, [0.5] * 6 + [0.5] * 6, clusters=["a"] * 6 + ["b"] * 6)
    clustered = difference_estimate(data, use_clusters=True)
    naive = difference_estimate(data, use_clusters=False)
    assert clustered.method == "paired-clustered"
    assert clustered.n == 2
    assert naive.n == 12
    assert clustered.se is not None and naive.se is not None
    assert clustered.se > naive.se


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------


def test_benjamini_hochberg_matches_a_hand_worked_example() -> None:
    """P = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205], m=8, q=0.05.

    The largest k with p_(k) <= k*q/m is k=5 (0.042 <= 5*0.05/8 = 0.03125 is false;
    0.041 <= 0.025 false; 0.039 <= 0.01875 false; 0.008 <= 0.0125 true at k=2), so BH rejects
    the first two.
    """
    p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
    result = benjamini_hochberg(p_values, q=0.05)
    assert result.rejected[:2] == (True, True)
    assert result.n_rejected == 2
    assert result.adjusted[0] == pytest.approx(0.008, abs=1e-9)
    assert result.adjusted[1] == pytest.approx(0.032, abs=1e-9)


def test_adjusted_p_values_are_monotone_and_bounded() -> None:
    result = benjamini_hochberg([0.9, 0.01, 0.5, 0.001, 0.2])
    ordered = sorted(zip(result.raw, result.adjusted, strict=True))
    adjusted = [value for _, value in ordered]
    assert adjusted == sorted(adjusted), "adjusted p-values must not decrease with raw p"
    assert all(0.0 <= value <= 1.0 for value in result.adjusted)


def test_bh_rejects_a_superset_of_bonferroni() -> None:
    p_values = [0.001, 0.004, 0.01, 0.03, 0.2, 0.5]
    bh = benjamini_hochberg(p_values, q=0.05)
    bonf = bonferroni(p_values, alpha=0.05)
    for index in range(len(p_values)):
        if bonf.rejected[index]:
            assert bh.rejected[index], "BH must reject everything Bonferroni does"


def test_bh_preserves_input_order() -> None:
    result = benjamini_hochberg([0.5, 0.001], names=["late", "early"])
    lookup = result.by_name()
    assert lookup["early"][0] == 0.001
    assert lookup["late"][0] == 0.5


def test_bh_on_an_empty_family_is_empty() -> None:
    assert benjamini_hochberg([]).n_rejected == 0


def test_bh_with_all_nulls_rejects_nothing() -> None:
    assert benjamini_hochberg([0.4, 0.6, 0.8, 0.99]).n_rejected == 0


def test_bh_with_a_single_metric_is_the_raw_p_value() -> None:
    result = benjamini_hochberg([0.03], q=0.05)
    assert result.adjusted[0] == pytest.approx(0.03)
    assert result.rejected[0]


def test_bh_handles_ties() -> None:
    result = benjamini_hochberg([0.02, 0.02, 0.02], q=0.05)
    assert result.n_rejected == 3
    assert len(set(result.adjusted)) == 1


def test_the_multiplicity_problem_is_quantified_for_the_report() -> None:
    assert expected_false_alarms(12, 0.05) == pytest.approx(0.6)
    assert family_wise_error(12, 0.05) == pytest.approx(0.4596, abs=1e-3)


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------


def test_power_rises_with_sample_size_and_effect() -> None:
    small = paired_power(effect=0.05, sigma_d=0.2, n=20)
    larger_n = paired_power(effect=0.05, sigma_d=0.2, n=200)
    larger_effect = paired_power(effect=0.15, sigma_d=0.2, n=20)
    assert larger_n > small
    assert larger_effect > small


def test_power_at_zero_effect_is_alpha() -> None:
    assert paired_power(effect=0.0, sigma_d=0.2, n=50, alpha=0.05) == pytest.approx(0.05, abs=1e-6)


def test_power_matches_a_closed_form_normal_approximation_at_large_n() -> None:
    """At n=5000 the t distribution is effectively normal, so the two must agree."""
    effect, sigma, n, alpha = 0.02, 0.2, 5000, 0.05
    ours = paired_power(effect=effect, sigma_d=sigma, n=n, alpha=alpha)
    lam = effect * math.sqrt(n) / sigma
    approx = float(scipy_stats.norm.sf(scipy_stats.norm.ppf(1 - alpha) - lam))
    assert ours == pytest.approx(approx, abs=0.01)


def test_mde_is_the_effect_reaching_the_target_power() -> None:
    mde = minimum_detectable_effect(sigma_d=0.2, n=50, alpha=0.05, power=0.8)
    assert paired_power(effect=mde, sigma_d=0.2, n=50) == pytest.approx(0.8, abs=0.01)


def test_mde_shrinks_as_the_suite_grows() -> None:
    assert minimum_detectable_effect(sigma_d=0.2, n=200) < minimum_detectable_effect(
        sigma_d=0.2, n=20
    )


def test_required_pairs_reaches_the_target_power() -> None:
    n = required_pairs(effect=0.03, sigma_d=0.15, alpha=0.05, power=0.8)
    assert paired_power(effect=0.03, sigma_d=0.15, n=n) >= 0.8
    assert paired_power(effect=0.03, sigma_d=0.15, n=n - 1) < 0.8


def test_an_undetectable_effect_reports_the_ceiling_rather_than_lying() -> None:
    assert required_pairs(effect=0.0, sigma_d=0.2) == 100_000


def test_noiseless_designs_detect_everything() -> None:
    assert paired_power(effect=0.01, sigma_d=0.0, n=10) == 1.0
    assert required_pairs(effect=0.01, sigma_d=0.0) == 2


def test_power_report_and_its_sentence() -> None:
    report = power_report(PowerInputs(sigma_d=0.2, n_pairs=30, margin=0.03))
    assert 0.0 <= report.achieved_power <= 1.0
    assert report.mde > 0.03, "a 30-task suite cannot detect a 3-point change at this noise"
    sentence = describe(report, "outcome.task_success", 0.03)
    assert "80% power" in sentence
    assert "configured margin" in sentence


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------


def test_pass_at_k_and_pass_hat_k_at_the_extremes() -> None:
    assert pass_at_k(4, 4, 1) == 1.0
    assert pass_at_k(0, 4, 1) == 0.0
    assert pass_hat_k(4, 4, 4) == 1.0
    assert pass_hat_k(3, 4, 4) == 0.0
    assert pass_hat_k(2, 4, 1) == pytest.approx(0.5)


def test_pass_hat_k_matches_the_combinatorial_definition() -> None:
    """c=3, K=4, k=2: C(3,2)/C(4,2) = 3/6 = 0.5."""
    assert pass_hat_k(3, 4, 2) == pytest.approx(0.5)
    assert pass_at_k(1, 4, 2) == pytest.approx(1 - math.comb(3, 2) / math.comb(4, 2))


def test_pass_hat_k_is_non_increasing_in_k() -> None:
    values = [pass_hat_k(3, 4, k) for k in range(1, 5)]
    assert values == sorted(values, reverse=True)


def test_the_unbiased_estimator_differs_from_the_naive_plug_in() -> None:
    """(3/4)^2 = 0.5625 but the unbiased estimate is 0.5 — the naive form is biased upward here."""
    assert naive_pass_hat_k(3, 4, 2) == pytest.approx(0.5625)
    assert pass_hat_k(3, 4, 2) == pytest.approx(0.5)


def test_estimating_beyond_the_repetitions_run_is_refused() -> None:
    with pytest.raises(ValueError, match="extrapolation"):
        pass_hat_k(4, 4, 5)


def test_flake_rate_counts_tasks_that_sometimes_pass() -> None:
    outcomes = outcomes_from_scores(
        {"a": [1.0, 1.0], "b": [1.0, 0.0], "c": [0.0, 0.0], "d": [0.0, 1.0]}
    )
    assert flake_rate(outcomes) == pytest.approx(0.5)
    assert flake_rate([]) == 0.0


def test_reliability_report_shows_the_pass_hat_k_decay() -> None:
    scores = {f"t{i}": [1.0, 1.0, 1.0, 0.0] for i in range(20)}
    report = reliability_report("outcome.task_success", scores, resamples=400)

    assert report.k_max == 4
    curve = [point.pass_hat_k.value for point in report.curve]
    assert curve[0] == pytest.approx(0.75)
    assert curve == sorted(curve, reverse=True)
    assert curve[-1] == pytest.approx(0.0), "no task passes all four repetitions"
    assert report.flake_rate.value == pytest.approx(1.0)
    assert len(report.flakiest_tasks) == 10


def test_a_perfectly_reliable_agent_has_a_flat_curve() -> None:
    scores = {f"t{i}": [1.0] * 4 for i in range(10)}
    report = reliability_report("outcome.task_success", scores, resamples=200)
    assert all(point.pass_hat_k.value == pytest.approx(1.0) for point in report.curve)
    assert report.flake_rate.value == 0.0
    assert report.flakiest_tasks == []


def test_score_variance_surfaces_nondeterminism_behind_equal_means() -> None:
    from agentgate.stats import score_variance

    steady = {f"t{i}": [0.5, 0.5, 0.5, 0.5] for i in range(10)}
    erratic = {f"t{i}": [0.0, 1.0, 0.0, 1.0] for i in range(10)}
    assert score_variance(steady) == pytest.approx(0.0)
    assert score_variance(erratic) > 0.3


# ---------------------------------------------------------------------------
# Variance decomposition and judge propagation
# ---------------------------------------------------------------------------


def test_decomposition_attributes_variance_to_tasks_when_reps_agree() -> None:
    scores = {f"t{i}": [i / 10.0] * 4 for i in range(10)}
    decomposition = decompose(scores)
    assert decomposition is not None
    assert decomposition.within_task_var == pytest.approx(0.0)
    assert decomposition.icc == pytest.approx(1.0)
    assert "more tasks" in decomposition.recommendation


def test_decomposition_attributes_variance_to_reps_when_tasks_agree() -> None:
    rng = np.random.default_rng(5)
    scores = {f"t{i}": rng.normal(0.5, 0.3, 6).tolist() for i in range(30)}
    decomposition = decompose(scores)
    assert decomposition is not None
    assert decomposition.icc < 0.3
    assert "More repetitions" in decomposition.recommendation


def test_decomposition_needs_tasks_and_repetitions() -> None:
    assert decompose({"t0": [0.5, 0.6]}) is None
    assert decompose({"t0": [0.5], "t1": [0.6]}) is None


def test_judge_variance_widens_the_standard_error() -> None:
    means = [0.5, 0.6, 0.4, 0.55, 0.45, 0.5]
    plain = propagate_judge_variance(means, [], n_judge_samples=3)
    inflated = propagate_judge_variance(means, [0.04] * 6, n_judge_samples=3)
    assert inflated > plain
    # SE^2 = var/n + mean_judge_var/(J*K*n) = plain^2 + 0.04/(3*1*6)
    assert inflated == pytest.approx(math.sqrt(plain**2 + 0.04 / 18), abs=1e-9)


def test_more_judge_samples_shrink_the_judge_term() -> None:
    means = [0.5, 0.6, 0.4, 0.55]
    assert propagate_judge_variance(means, [0.04] * 4, n_judge_samples=9) < (
        propagate_judge_variance(means, [0.04] * 4, n_judge_samples=3)
    )
