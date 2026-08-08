"""Property-based invariants for the statistics engine (H2).

Laws that must hold for every input, not just the ones a test author thought of. The Monte-Carlo
suite checks that the engine is *calibrated*; these check that it is never *incoherent*.
"""

from __future__ import annotations

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from agentgate.stats import (
    benjamini_hochberg,
    bonferroni,
    clt_interval,
    minimum_detectable_effect,
    paired_power,
    pass_at_k,
    pass_hat_k,
    permutation_test,
    required_pairs,
    wilson_interval,
)

probabilities = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
p_value_lists = st.lists(probabilities, min_size=1, max_size=25)
scores = st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(successes=st.integers(min_value=0, max_value=500), extra=st.integers(0, 500))
def test_wilson_stays_in_the_unit_interval_and_contains_the_point_estimate(
    successes: int, extra: int
) -> None:
    trials = successes + extra
    assume(trials > 0)
    estimate = wilson_interval(successes, trials)
    assert estimate.ci_low is not None and estimate.ci_high is not None
    assert 0.0 <= estimate.ci_low <= estimate.ci_high <= 1.0
    assert estimate.ci_low <= estimate.value <= estimate.ci_high


@settings(max_examples=200, deadline=None)
@given(successes=st.integers(1, 200), extra=st.integers(1, 200))
def test_wilson_narrows_as_evidence_accumulates(successes: int, extra: int) -> None:
    """Doubling the sample at the same proportion must not widen the interval."""
    trials = successes + extra
    small = wilson_interval(successes, trials)
    large = wilson_interval(successes * 2, trials * 2)
    assert small.half_width is not None and large.half_width is not None
    assert large.half_width <= small.half_width + 1e-12


@settings(max_examples=200, deadline=None)
@given(values=st.lists(scores, min_size=2, max_size=60))
def test_clt_interval_brackets_its_own_mean(values: list[float]) -> None:
    estimate = clt_interval(values)
    assert estimate.ci_low is not None and estimate.ci_high is not None
    assert estimate.ci_low <= estimate.value <= estimate.ci_high


# ---------------------------------------------------------------------------
# Multiplicity
# ---------------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(p_values=p_value_lists)
def test_adjusted_p_values_stay_in_range_and_never_shrink(p_values: list[float]) -> None:
    result = benjamini_hochberg(p_values)
    for raw, adjusted in zip(result.raw, result.adjusted, strict=True):
        assert 0.0 <= adjusted <= 1.0
        assert adjusted >= raw - 1e-12, "an adjustment must never make a p-value smaller"


@settings(max_examples=300, deadline=None)
@given(p_values=p_value_lists)
def test_bh_rejects_a_superset_of_bonferroni(p_values: list[float]) -> None:
    """The relationship that makes BH the right choice for a gate: strictly more sensitive."""
    bh = benjamini_hochberg(p_values, q=0.05)
    bonf = bonferroni(p_values, alpha=0.05)
    for index in range(len(p_values)):
        if bonf.rejected[index]:
            assert bh.rejected[index]


@settings(max_examples=200, deadline=None)
@given(p_values=p_value_lists)
def test_bh_adjusted_p_is_monotone_in_the_raw_p(p_values: list[float]) -> None:
    result = benjamini_hochberg(p_values)
    from itertools import pairwise

    pairs = sorted(zip(result.raw, result.adjusted, strict=True))
    adjusted = [value for _, value in pairs]
    assert all(a <= b + 1e-12 for a, b in pairwise(adjusted))


@settings(max_examples=200, deadline=None)
@given(p_values=p_value_lists, q=st.floats(0.01, 0.5))
def test_rejection_and_adjusted_threshold_agree(p_values: list[float], q: float) -> None:
    """Rejection must agree with the adjusted-p threshold, or the report contradicts itself.

    Compared with a tolerance on both sides: the equivalence is exact in real arithmetic and
    ULP-sensitive in floating point, and pinning the ULP would test the FPU, not the method.
    """
    result = benjamini_hochberg(p_values, q=q)
    for adjusted, rejected in zip(result.adjusted, result.rejected, strict=True):
        if adjusted <= q - 1e-12:
            assert rejected, "an adjusted p clearly below q must be rejected"
        if adjusted >= q + 1e-12:
            assert not rejected, "an adjusted p clearly above q must not be rejected"


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(trials=st.integers(1, 12), successes=st.integers(0, 12))
def test_reliability_estimators_stay_in_the_unit_interval(trials: int, successes: int) -> None:
    assume(successes <= trials)
    for k in range(1, trials + 1):
        assert 0.0 <= pass_hat_k(successes, trials, k) <= 1.0
        assert 0.0 <= pass_at_k(successes, trials, k) <= 1.0


@settings(max_examples=300, deadline=None)
@given(trials=st.integers(1, 12), successes=st.integers(0, 12))
def test_pass_hat_k_is_non_increasing_and_pass_at_k_non_decreasing(
    trials: int, successes: int
) -> None:
    assume(successes <= trials)
    hat = [pass_hat_k(successes, trials, k) for k in range(1, trials + 1)]
    at = [pass_at_k(successes, trials, k) for k in range(1, trials + 1)]
    assert hat == sorted(hat, reverse=True), "requiring more successes cannot get easier"
    assert at == sorted(at), "allowing more attempts cannot get harder"


@settings(max_examples=200, deadline=None)
@given(trials=st.integers(1, 12), successes=st.integers(0, 12))
def test_pass_at_one_and_pass_hat_one_are_both_the_success_rate(
    trials: int, successes: int
) -> None:
    assume(successes <= trials)
    rate = successes / trials
    assert pass_at_k(successes, trials, 1) == pytest.approx(rate)
    assert pass_hat_k(successes, trials, 1) == pytest.approx(rate)


# ---------------------------------------------------------------------------
# Permutation
# ---------------------------------------------------------------------------


@settings(max_examples=40, deadline=None)
@given(
    differences=st.lists(
        st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False), min_size=2, max_size=20
    )
)
def test_permutation_p_is_bounded_by_its_own_resolution(differences: list[float]) -> None:
    """A permutation p-value cannot claim more evidence than B resamples can supply."""
    iterations = 200
    result = permutation_test(differences, iterations=iterations, seed=1)
    assert 1.0 / (iterations + 1) <= result.p_one_sided <= 1.0


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------


@settings(max_examples=200, deadline=None)
@given(
    effect=st.floats(0.001, 1.0),
    sigma=st.floats(0.001, 1.0),
    n=st.integers(2, 500),
)
def test_power_is_a_probability_and_rises_with_sample_size(
    effect: float, sigma: float, n: int
) -> None:
    power = paired_power(effect=effect, sigma_d=sigma, n=n)
    assert 0.0 <= power <= 1.0
    assert paired_power(effect=effect, sigma_d=sigma, n=n * 2) >= power - 1e-9


@settings(max_examples=100, deadline=None)
@given(sigma=st.floats(0.01, 1.0), n=st.integers(5, 300))
def test_the_mde_is_exactly_the_effect_reaching_target_power(sigma: float, n: int) -> None:
    mde = minimum_detectable_effect(sigma_d=sigma, n=n, power=0.8)
    assert paired_power(effect=mde, sigma_d=sigma, n=n) >= 0.8 - 1e-3


@settings(max_examples=100, deadline=None)
@given(effect=st.floats(0.01, 0.5), sigma=st.floats(0.01, 0.5))
def test_required_pairs_actually_reaches_the_target(effect: float, sigma: float) -> None:
    n = required_pairs(effect=effect, sigma_d=sigma, power=0.8)
    assume(n < 100_000)
    assert paired_power(effect=effect, sigma_d=sigma, n=n) >= 0.8 - 1e-3
