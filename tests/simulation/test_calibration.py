"""Monte-Carlo validation of the statistics engine (H3) — the credibility centrepiece.

A gate that claims a 5% false-alarm rate must *have* a 5% false-alarm rate. These simulations
check that against synthetic agents whose truth is known by construction:

1. **Type-I error calibration** — identical systems must produce rejections at the nominal rate.
2. **Coverage** — 95% intervals must contain the truth 93-97% of the time, and naive SEs must
   demonstrably under-cover under clustering.
3. **Power realism** — the engine's predicted power must match the empirical detection rate.
4. **pass^k unbiasedness** — the combinatorial estimator must converge to the true value where
   the naive plug-in does not.
5. **Pairing benefit** — variance reduction must grow with the paired correlation, reproducing
   E3's central claim on data where the correlation is set rather than observed.

Marked ``slow``: run weekly and on any change under ``stats/``.
"""

from __future__ import annotations

import numpy as np
import pytest

from agentgate.stats import (
    benjamini_hochberg,
    clt_interval,
    clustered_interval,
    naive_pass_hat_k,
    paired_power,
    paired_t_test,
    pass_hat_k,
    run_paired_test,
    wilson_interval,
)
from tests.simulation.synthetic import (
    BernoulliAgents,
    GaussianAgents,
    SuiteDesign,
    covers,
    paired_differences,
    task_level,
)

pytestmark = pytest.mark.slow

REPLICATIONS = 2000
"""Per the spec's H3: at least 2,000 simulated experiments per case."""

TYPE_I_BAND = (0.03, 0.07)
COVERAGE_BAND = (0.93, 0.97)
POWER_TOLERANCE = 0.05


def binomial_halfwidth(rate: float, n: int) -> float:
    """Monte-Carlo error bar on an estimated rate, so tolerances are not arbitrary."""
    return 1.96 * float(np.sqrt(max(rate * (1 - rate), 1e-6) / n))


# ---------------------------------------------------------------------------
# 1. Type-I error calibration
# ---------------------------------------------------------------------------


def test_identical_systems_reject_at_the_nominal_rate() -> None:
    """With no true effect, a one-sided test at alpha=0.05 must reject about 5% of the time."""
    rng = np.random.default_rng(20260101)
    agents = GaussianAgents(design=SuiteDesign(n_tasks=40, k_reps=4), effect=0.0)
    rejections = 0
    for _ in range(REPLICATIONS):
        baseline, candidate, _ = agents.draw(rng)
        differences = paired_differences(baseline, candidate)
        result = paired_t_test(differences, alternative="less")
        rejections += result.p_one_sided < 0.05

    rate = rejections / REPLICATIONS
    assert TYPE_I_BAND[0] <= rate <= TYPE_I_BAND[1], (
        f"type-I error {rate:.4f} outside {TYPE_I_BAND}"
    )


def test_the_margin_makes_a_small_true_regression_non_significant() -> None:
    """A candidate worse by less than the margin must not be flagged as a regression."""
    rng = np.random.default_rng(4242)
    margin = 0.05
    agents = GaussianAgents(design=SuiteDesign(n_tasks=40, k_reps=4), effect=-0.01)
    rejections = 0
    for _ in range(REPLICATIONS):
        baseline, candidate, _ = agents.draw(rng)
        differences = paired_differences(baseline, candidate)
        result = paired_t_test(differences, shift=margin, alternative="less")
        rejections += result.p_one_sided < 0.05

    rate = rejections / REPLICATIONS
    assert rate < 0.01, (
        f"a 1-point drop inside a 5-point margin was called a regression {rate:.1%} of the time"
    )


def test_auto_selected_tests_stay_calibrated_on_non_normal_data() -> None:
    """Test *selection* must not itself inflate the false-alarm rate."""
    rng = np.random.default_rng(777)
    rejections = 0
    replications = 1000
    for _ in range(replications):
        # Symmetric but heavy-tailed differences: null is true, shape is not normal.
        differences = (rng.standard_t(3, size=40) * 0.05).tolist()
        result = run_paired_test(differences, alternative="less")
        rejections += result.p_one_sided < 0.05

    rate = rejections / replications
    half = binomial_halfwidth(0.05, replications)
    assert 0.05 - 3 * half <= rate <= 0.05 + 3 * half, f"selection-inflated type-I error {rate:.4f}"


def test_fdr_holds_the_false_discovery_rate_across_a_metric_family() -> None:
    """Twelve all-null metrics at q=0.05: the family-wise rejection rate must stay near q."""
    rng = np.random.default_rng(31337)
    families_with_any_rejection = 0
    replications = 2000
    for _ in range(replications):
        p_values = rng.uniform(0.0, 1.0, 12).tolist()
        adjustment = benjamini_hochberg(p_values, q=0.05)
        families_with_any_rejection += adjustment.n_rejected > 0

    rate = families_with_any_rejection / replications
    assert rate <= 0.07, f"BH let {rate:.1%} of all-null families produce a rejection"


def test_uncorrected_testing_would_false_alarm_far_more_often() -> None:
    """The number that justifies the correction existing at all (C3)."""
    rng = np.random.default_rng(999)
    naive_alarms = 0
    replications = 2000
    for _ in range(replications):
        p_values = rng.uniform(0.0, 1.0, 12)
        naive_alarms += bool(np.any(p_values < 0.05))

    naive_rate = naive_alarms / replications
    assert naive_rate > 0.35, "expected roughly 46% family-wise error without correction"


# ---------------------------------------------------------------------------
# 2. Coverage
# ---------------------------------------------------------------------------


def test_clt_intervals_cover_at_the_nominal_rate() -> None:
    rng = np.random.default_rng(2024)
    truth = 0.7
    agents = GaussianAgents(
        design=SuiteDesign(n_tasks=50, k_reps=4), baseline_mean=truth, cluster_sd=0.0
    )
    hits = 0
    for _ in range(REPLICATIONS):
        baseline, _, _ = agents.draw(rng)
        hits += covers(clt_interval(task_level(baseline)), truth)

    coverage = hits / REPLICATIONS
    assert COVERAGE_BAND[0] <= coverage <= COVERAGE_BAND[1], f"CLT coverage {coverage:.4f}"


def test_wilson_intervals_cover_even_at_extreme_proportions() -> None:
    """The regime where the normal approximation fails and Wilson is why we use it."""
    rng = np.random.default_rng(606)
    truth, n = 0.92, 40
    hits = 0
    for _ in range(REPLICATIONS):
        successes = int(rng.binomial(n, truth))
        estimate = wilson_interval(successes, n)
        assert estimate.ci_low is not None and estimate.ci_high is not None
        hits += estimate.ci_low <= truth <= estimate.ci_high

    coverage = hits / REPLICATIONS
    assert coverage >= 0.93, f"Wilson coverage {coverage:.4f} at p=0.92"


def test_naive_intervals_under_cover_under_clustering_and_clustered_ones_repair_it() -> None:
    """Reproduces E3's warning: ignoring cluster structure understates uncertainty."""
    rng = np.random.default_rng(8080)
    truth = 0.7
    agents = GaussianAgents(
        design=SuiteDesign(n_tasks=60, k_reps=2, n_clusters=6),
        baseline_mean=truth,
        task_sd=0.05,
        rep_sd=0.05,
        cluster_sd=0.20,
    )
    naive_hits = clustered_hits = 0
    inflations = []
    replications = 1000
    for _ in range(replications):
        baseline, _, clusters = agents.draw(rng)
        task_ids = sorted(baseline)
        values = task_level(baseline)
        ids = [clusters[task_id] for task_id in task_ids]

        naive = clt_interval(values)
        grouped = clustered_interval(values, ids)
        naive_hits += covers(naive, truth)
        clustered_hits += covers(grouped, truth)
        if naive.se and grouped.se:
            inflations.append(grouped.se / naive.se)

    naive_coverage = naive_hits / replications
    clustered_coverage = clustered_hits / replications
    assert naive_coverage < 0.80, (
        f"naive coverage {naive_coverage:.2f} should collapse under strong clustering"
    )
    assert clustered_coverage > naive_coverage + 0.10, (
        f"clustered coverage {clustered_coverage:.2f} must repair naive {naive_coverage:.2f}"
    )
    assert float(np.median(inflations)) > 2.0, "SE inflation should be large in this regime"


def test_paired_difference_intervals_cover_the_true_effect() -> None:
    rng = np.random.default_rng(5150)
    effect = -0.04
    agents = GaussianAgents(design=SuiteDesign(n_tasks=50, k_reps=4), effect=effect)
    hits = 0
    for _ in range(REPLICATIONS):
        baseline, candidate, _ = agents.draw(rng)
        hits += covers(clt_interval(paired_differences(baseline, candidate)), effect)

    coverage = hits / REPLICATIONS
    assert COVERAGE_BAND[0] <= coverage <= COVERAGE_BAND[1], f"paired coverage {coverage:.4f}"


# ---------------------------------------------------------------------------
# 3. Power realism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("effect", [-0.05, -0.10])
def test_empirical_detection_rate_matches_predicted_power(effect: float) -> None:
    """The engine's power prediction must match what actually happens (C4)."""
    rng = np.random.default_rng(1234)
    design = SuiteDesign(n_tasks=40, k_reps=4)
    agents = GaussianAgents(design=design, effect=effect)

    detections = 0
    sigmas = []
    for _ in range(REPLICATIONS):
        baseline, candidate, _ = agents.draw(rng)
        differences = paired_differences(baseline, candidate)
        sigmas.append(float(np.std(differences, ddof=1)))
        detections += paired_t_test(differences, alternative="less").p_one_sided < 0.05

    empirical = detections / REPLICATIONS
    predicted = paired_power(
        effect=abs(effect), sigma_d=float(np.mean(sigmas)), n=design.n_tasks, alpha=0.05
    )
    assert abs(empirical - predicted) <= POWER_TOLERANCE, (
        f"empirical power {empirical:.3f} vs predicted {predicted:.3f}"
    )


def test_power_prediction_is_monotone_in_sample_size_empirically() -> None:
    rng = np.random.default_rng(60606)
    results = {}
    for n_tasks in (20, 80):
        agents = GaussianAgents(design=SuiteDesign(n_tasks=n_tasks, k_reps=4), effect=-0.05)
        detections = sum(
            paired_t_test(paired_differences(*agents.draw(rng)[:2]), alternative="less").p_one_sided
            < 0.05
            for _ in range(800)
        )
        results[n_tasks] = detections / 800
    assert results[80] > results[20]


# ---------------------------------------------------------------------------
# 4. pass^k unbiasedness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [0.5, 0.75, 0.9])
@pytest.mark.parametrize("k", [2, 3])
def test_pass_hat_k_is_mean_unbiased(p: float, k: int) -> None:
    """E[C(c,k)/C(K,k)] must equal p^k; the naive plug-in is biased upward."""
    rng = np.random.default_rng(int(p * 1000) + k)
    trials = 4
    replications = 20000

    successes = rng.binomial(trials, p, replications)
    unbiased = np.array([pass_hat_k(int(c), trials, k) for c in successes])
    naive = np.array([naive_pass_hat_k(int(c), trials, k) for c in successes])

    truth = p**k
    half = binomial_halfwidth(truth, replications) * 2
    assert abs(float(unbiased.mean()) - truth) <= half, (
        f"pass^{k} estimator mean {unbiased.mean():.4f} vs truth {truth:.4f}"
    )
    assert abs(float(naive.mean()) - truth) > abs(float(unbiased.mean()) - truth), (
        "the naive plug-in should be measurably more biased than the combinatorial estimator"
    )


def test_pass_at_k_is_mean_unbiased() -> None:
    from agentgate.stats import pass_at_k

    rng = np.random.default_rng(4321)
    p, trials, k, replications = 0.6, 4, 2, 20000
    successes = rng.binomial(trials, p, replications)
    estimates = np.array([pass_at_k(int(c), trials, k) for c in successes])
    truth = 1.0 - (1.0 - p) ** k
    assert abs(float(estimates.mean()) - truth) <= binomial_halfwidth(truth, replications) * 2


def test_pass_hat_k_decays_the_way_tau_bench_describes() -> None:
    """A 75%-success agent should fall well below 50% by pass^3 (E2)."""
    rng = np.random.default_rng(2468)
    agents = BernoulliAgents(design=SuiteDesign(n_tasks=200, k_reps=4), baseline_p=0.75)
    baseline, _, _ = agents.draw(rng)

    curve = []
    for k in range(1, 5):
        values = [pass_hat_k(int(sum(reps)), len(reps), k) for reps in baseline.values()]
        curve.append(float(np.mean(values)))

    assert curve[0] == pytest.approx(0.75, abs=0.05)
    assert curve[2] < 0.50, f"pass^3 was {curve[2]:.3f}; the decay is the whole point"
    assert curve == sorted(curve, reverse=True)


# ---------------------------------------------------------------------------
# 5. Pairing benefit (E3's core claim)
# ---------------------------------------------------------------------------


def test_pairing_variance_reduction_grows_with_correlation() -> None:
    """Reproduces E3 on synthetic data: the higher the correlation, the more pairing buys."""
    rng = np.random.default_rng(13579)
    ratios: dict[float, float] = {}
    for rho in (0.0, 0.3, 0.5, 0.7):
        agents = GaussianAgents(
            design=SuiteDesign(n_tasks=60, k_reps=2, pairing_correlation=rho), effect=0.0
        )
        paired_variances = []
        unpaired_variances = []
        for _ in range(400):
            baseline, candidate, _ = agents.draw(rng)
            base = np.array(task_level(baseline))
            cand = np.array(task_level(candidate))
            paired_variances.append(float(np.var(cand - base, ddof=1)))
            unpaired_variances.append(float(np.var(base, ddof=1) + np.var(cand, ddof=1)))
        ratios[rho] = float(np.mean(paired_variances)) / float(np.mean(unpaired_variances))

    assert ratios[0.7] < ratios[0.5] < ratios[0.3] < ratios[0.0], ratios
    assert ratios[0.0] == pytest.approx(1.0, abs=0.15), "uncorrelated pairing buys nothing"
    assert ratios[0.5] < 0.85, "pairing at r=0.5 should cut a real share of the variance"


def test_pairing_raises_power_at_the_same_sample_size() -> None:
    """The practical consequence: the same suite detects smaller effects when pairing works."""
    rng = np.random.default_rng(24680)
    detections: dict[float, float] = {}
    for rho in (0.0, 0.7):
        agents = GaussianAgents(
            design=SuiteDesign(n_tasks=40, k_reps=2, pairing_correlation=rho), effect=-0.05
        )
        hits = sum(
            paired_t_test(paired_differences(*agents.draw(rng)[:2]), alternative="less").p_one_sided
            < 0.05
            for _ in range(800)
        )
        detections[rho] = hits / 800

    assert detections[0.7] > detections[0.0] + 0.05, detections
