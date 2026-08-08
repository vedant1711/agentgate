"""The gate decision rule (C3) — the project's core novelty.

A naive gate asks "did the number go down by more than 3%?". That question has no answer at
n = 30, because the number moves by more than that from sampling noise alone. AgentGate asks a
different question, and can give three different answers to it:

* **REGRESSION** — a one-sided paired test rejects ``H0: Delta >= -delta``. The candidate is
  worse by more than the margin, and the evidence supports saying so.
* **PASS** — non-inferiority established: the test rejects ``H0: Delta <= -delta``, so the
  confidence bound sits above the margin. The candidate is *proven fine*, not merely unproven
  bad.
* **INCONCLUSIVE / UNDERPOWERED** — neither rejects. If the suite could not have detected the
  configured margin at the target power, the verdict says ``UNDERPOWERED`` rather than passing.

That third answer is the point. A naive gate is silent exactly when it knows least, and its
silence is indistinguishable from a pass. Saying "this suite cannot tell" is more useful than a
green tick that means nothing.

Two things bypass the statistics entirely. Safety tripwires fail the gate on any *new* failure —
"the increase in PII leakage was not statistically significant" is not a sentence anyone should
read after an incident. And a judge-backed metric whose judge has not cleared the kappa floor may
be reported but may not gate, because a gate on an unvalidated instrument measures the
instrument.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from agentgate.schemas.common import Verdict, stable_hash
from agentgate.schemas.policy import GatedMetric, GatePolicy
from agentgate.schemas.results import (
    ComparisonResult,
    GateVerdict,
    JudgeHealth,
    MetricComparison,
    PairedTestResult,
    PowerReport,
    SafetyFinding,
)
from agentgate.stats.multiplicity import benjamini_hochberg
from agentgate.stats.paired import run_paired_test
from agentgate.stats.power import PowerInputs, power_report

EXIT_PASS = 0
EXIT_REGRESSION = 1
EXIT_SAFETY = 2
EXIT_UNDERPOWERED = 3
EXIT_ERROR = 4


@dataclass(slots=True)
class MetricRuling:
    """The gate's decision on one metric, with everything it rested on."""

    metric: str
    verdict: Verdict
    comparison: MetricComparison | None
    margin: float
    p_regression: float | None = None
    p_adjusted: float | None = None
    p_noninferiority: float | None = None
    achieved_power: float | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def blocks(self) -> bool:
        """True when this ruling should fail the build on its own."""
        return self.verdict in (Verdict.REGRESSION, Verdict.SAFETY_FAIL)


def resolve_margin(entry: GatedMetric, comparison: MetricComparison) -> float:
    """Turn a policy margin into an absolute, direction-normalised delta.

    A ratio margin such as ``1.25`` on latency means "up to 25% slower is acceptable". That is a
    *relative* allowance, so it becomes an absolute one against the baseline mean: a suite whose
    baseline latency is 400ms tolerates 100ms, and a faster baseline tolerates less. Expressing
    it any other way would make the same policy mean different things on different suites.

    Args:
        entry: The policy entry.
        comparison: The metric's paired comparison, for the baseline scale.

    Returns:
        A non-negative absolute margin in direction-normalised units.
    """
    if entry.margin is not None:
        return abs(entry.margin)
    ratio = entry.margin_ratio or 1.0
    baseline = abs(comparison.baseline.value)
    return abs(ratio - 1.0) * baseline


def new_safety_failures(
    comparison: ComparisonResult, tripwires: Sequence[str]
) -> list[SafetyFinding]:
    """Safety failures the candidate introduced, restricted to the configured tripwires."""
    watched = set(tripwires)
    return [
        finding
        for finding in comparison.safety_findings
        if finding.metric in watched and finding.is_new
    ]


def judge_blocked(metric: str, health: JudgeHealth | None, policy: GatePolicy) -> str | None:
    """Return a reason when a judge-backed metric must not gate (D4).

    Args:
        metric: Metric name.
        health: Judge health panel, when the run had one.
        policy: Gate policy carrying the kappa floor.

    Returns:
        A human-readable reason, or ``None`` when the metric may gate.
    """
    if not metric.startswith(("judge.", "rag.faithfulness", "rag.hallucination")):
        return None
    if health is None:
        return (
            f"{metric} is judge-backed but this run has no judge-health panel; a gate on an "
            f"unvalidated instrument measures the instrument (D4)"
        )
    if health.cohens_kappa is None:
        return (
            f"{metric} is judge-backed and the judge has no human calibration; it may be "
            f"reported but must not gate (D4)"
        )
    if health.cohens_kappa < policy.require_judge_kappa:
        return (
            f"{metric} is judge-backed and its judge scores kappa={health.cohens_kappa:.2f}, "
            f"below the {policy.require_judge_kappa} floor; reported but not gated (D4)"
        )
    return None


def tests_at_margin(
    comparison: MetricComparison, *, margin: float, policy: GatePolicy
) -> tuple[PairedTestResult, PairedTestResult, PowerReport | None]:
    """Recompute both one-sided tests and the power report at ``margin``.

    The gate must never read a p-value computed for a *different* margin. Comparisons are built
    once, with whatever margins the policy declared at the time; but a verdict can be re-rendered
    later at a looser margin, a stricter alpha, or a different FDR level — and the demo's margin
    slider does exactly that. So the tests are recomputed here from the stored analysis units,
    which makes the verdict and its evidence agree by construction.

    Args:
        comparison: The paired comparison, carrying its analysis units.
        margin: The resolved absolute margin.
        policy: Gate policy, for the normality threshold and permutation budget.

    Returns:
        ``(regression test, non-inferiority test, power report)``. Falls back to the stored
        tests when the comparison predates ``analysis_units`` or has too few observations.
    """
    units = comparison.analysis_units
    if len(units) < 2:
        return comparison.regression_test, comparison.noninferiority_test, comparison.power

    regression = run_paired_test(
        units,
        shift=margin,
        alternative="less",
        normality_alpha=policy.normality_alpha,
        iterations=policy.permutation_iters,
    )
    noninferiority = run_paired_test(
        units,
        shift=margin,
        alternative="greater",
        normality_alpha=policy.normality_alpha,
        iterations=policy.permutation_iters,
    )
    sigma_d = float(np.std(units, ddof=1))
    power = power_report(
        PowerInputs(
            sigma_d=sigma_d,
            n_pairs=len(units),
            margin=margin,
            alpha=policy.alpha,
            target=policy.power_target,
        )
    )
    return regression, noninferiority, power


def rule_on_metric(
    entry: GatedMetric,
    comparison: MetricComparison | None,
    *,
    adjusted_p: float | None,
    policy: GatePolicy,
    health: JudgeHealth | None,
) -> MetricRuling:
    """Apply the C3 decision rule to one gated metric."""
    name = entry.resolved_name
    if comparison is None:
        return MetricRuling(
            metric=name,
            verdict=Verdict.SKIPPED,
            comparison=None,
            margin=entry.margin or 0.0,
            reasons=[f"{name} was not scored in both runs, so it could not be compared"],
        )

    margin = resolve_margin(entry, comparison)
    reasons: list[str] = []

    blocked = judge_blocked(entry.metric, health, policy)
    if blocked:
        reasons.append(blocked)
        return MetricRuling(
            metric=name,
            verdict=Verdict.SKIPPED,
            comparison=comparison,
            margin=margin,
            reasons=reasons,
        )

    if comparison.n_pairs < policy.min_pairs:
        reasons.append(
            f"only {comparison.n_pairs} paired task(s), below the configured minimum of "
            f"{policy.min_pairs}; no verdict is claimed"
        )
        return MetricRuling(
            metric=name,
            verdict=Verdict.UNDERPOWERED,
            comparison=comparison,
            margin=margin,
            reasons=reasons,
        )

    regression, noninferiority, power_report_at_margin = tests_at_margin(
        comparison, margin=margin, policy=policy
    )
    p_regression = regression.p_one_sided
    p_noninferiority = noninferiority.p_one_sided
    effective_p = adjusted_p if adjusted_p is not None else p_regression
    power = power_report_at_margin.achieved_power if power_report_at_margin else None
    if regression.selection_reason:
        reasons.append(f"test: {regression.test} — {regression.selection_reason}")

    if effective_p <= policy.fdr_q:
        reasons.append(
            f"regression established: {name} fell by {abs(comparison.delta.value):.4g}, beyond "
            f"the {margin:.4g} margin (adjusted p={effective_p:.4g} <= q={policy.fdr_q})"
        )
        verdict = Verdict.REGRESSION
    elif p_noninferiority <= policy.alpha:
        reasons.append(
            f"non-inferiority established: the confidence bound on the change sits above "
            f"-{margin:.4g} (p={p_noninferiority:.4g} <= alpha={policy.alpha})"
        )
        verdict = Verdict.PASS
    elif power is not None and power < policy.power_target:
        needed = comparison.power.n_required if comparison.power else 0
        reasons.append(
            f"neither test rejects, and this suite has only {power:.0%} power to detect a "
            f"{margin:.4g} change (target {policy.power_target:.0%}). "
            f"About {needed} paired tasks would be needed. The gate cannot tell."
        )
        verdict = Verdict.UNDERPOWERED
    else:
        reasons.append(
            f"neither regression nor non-inferiority was established, despite adequate power "
            f"({power:.0%} for a {margin:.4g} change)"
            if power is not None
            else "neither regression nor non-inferiority was established"
        )
        verdict = Verdict.INCONCLUSIVE

    reasons.extend(comparison.notes)
    return MetricRuling(
        metric=name,
        verdict=verdict,
        comparison=comparison,
        margin=margin,
        p_regression=p_regression,
        p_adjusted=adjusted_p,
        p_noninferiority=p_noninferiority,
        achieved_power=power,
        reasons=reasons,
    )


@dataclass(slots=True)
class GateResult:
    """A verdict plus the per-metric rulings that produced it."""

    verdict: GateVerdict
    rulings: list[MetricRuling]
    comparison: ComparisonResult
    policy: GatePolicy

    def ruling(self, metric: str) -> MetricRuling | None:
        """Find one metric's ruling."""
        return next((item for item in self.rulings if item.metric == metric), None)

    @property
    def passed(self) -> bool:
        """True when the gate lets the change through."""
        return self.verdict.passed


def evaluate(
    comparison: ComparisonResult, policy: GatePolicy, *, health: JudgeHealth | None = None
) -> GateResult:
    """Render a gate verdict from a paired comparison and a policy.

    Order matters: safety first, then the FDR-corrected regression family, then non-inferiority,
    then the honesty check. Safety cannot be outvoted by statistics, and statistics cannot be
    outvoted by an underpowered suite's silence.

    Args:
        comparison: The paired analysis.
        policy: Gate configuration.
        health: Judge health panel, when the run had judge-backed metrics.

    Returns:
        The verdict and every ruling behind it.
    """
    judge_health = health if health is not None else comparison.judge_health
    entries = policy.enabled_metrics
    lookup = {item.metric: item for item in comparison.metrics}

    # Benjamini-Hochberg across the regression tests of the gated family (C3). Only metrics that
    # actually produce a test enter the family — including a skipped metric would inflate m and
    # make the correction quietly more lenient.
    testable = [
        (entry, lookup[entry.metric])
        for entry in entries
        if entry.metric in lookup and lookup[entry.metric].n_pairs >= policy.min_pairs
    ]
    # The FDR family is built from p-values recomputed at each metric's *resolved* margin, so
    # the correction applies to the same hypotheses the verdicts rest on.
    family_p = [
        tests_at_margin(item, margin=resolve_margin(entry, item), policy=policy)[0].p_one_sided
        for entry, item in testable
    ]
    adjustment = benjamini_hochberg(
        family_p,
        q=policy.fdr_q,
        names=[entry.resolved_name for entry, _ in testable],
    )
    adjusted_by_metric = dict(zip(adjustment.names, adjustment.adjusted, strict=True))

    rulings = [
        rule_on_metric(
            entry,
            lookup.get(entry.metric),
            adjusted_p=adjusted_by_metric.get(entry.resolved_name),
            policy=policy,
            health=judge_health,
        )
        for entry in entries
    ]

    safety = new_safety_failures(comparison, policy.safety_tripwires)
    warnings = list(comparison.warnings)
    if judge_health is not None:
        warnings.extend(judge_health.warnings)

    failing = [item.metric for item in rulings if item.verdict is Verdict.REGRESSION]
    underpowered = [item.metric for item in rulings if item.verdict is Verdict.UNDERPOWERED]

    if safety:
        overall, exit_code = Verdict.SAFETY_FAIL, EXIT_SAFETY
        summary = f"FAIL (SAFETY): {len(safety)} new safety failure(s) — " + ", ".join(
            sorted({finding.metric for finding in safety})
        )
    elif failing:
        overall, exit_code = Verdict.REGRESSION, EXIT_REGRESSION
        summary = f"FAIL: statistically significant regression in {', '.join(failing)}"
    elif underpowered and policy.underpowered_behavior == "fail":
        overall, exit_code = Verdict.UNDERPOWERED, EXIT_UNDERPOWERED
        summary = (
            f"FAIL (UNDERPOWERED, strict mode): {', '.join(underpowered)} could not be resolved "
            f"and the suite lacks the power to try"
        )
    elif underpowered:
        overall, exit_code = Verdict.UNDERPOWERED, EXIT_PASS
        summary = (
            f"PASS with warning: {', '.join(underpowered)} is UNDERPOWERED — this suite cannot "
            f"detect the configured margin, so its silence is not evidence of no regression"
        )
    else:
        overall, exit_code = Verdict.PASS, EXIT_PASS
        proven = [item.metric for item in rulings if item.verdict is Verdict.PASS]
        summary = (
            f"PASS: non-inferiority established for {', '.join(proven)}"
            if proven
            else "PASS: no gated metric regressed"
        )

    verdict = GateVerdict(
        comparison_id=comparison.comparison_id,
        created_at=datetime.now(UTC),
        verdict=overall,
        exit_code=exit_code,
        baseline_run_id=comparison.baseline_run_id,
        candidate_run_id=comparison.candidate_run_id,
        suite=comparison.suite,
        metric_verdicts={item.metric: item.verdict for item in rulings},
        failing_metrics=failing,
        underpowered_metrics=underpowered,
        safety_failures=safety,
        summary=summary,
        warnings=warnings,
        policy_hash=policy.policy_hash(),
    )
    return GateResult(verdict=verdict, rulings=rulings, comparison=comparison, policy=policy)


def naive_verdict(comparison: ComparisonResult, *, threshold: float = 0.03) -> dict[str, bool]:
    """The industry-standard fixed-threshold gate, for the report's side-by-side (E6).

    *"Fail if any metric drops more than 3%."* Implemented here purely so the report can show
    what it would have concluded on the same data — which is the argument for the whole project,
    made with numbers instead of assertion.

    Args:
        comparison: The paired analysis.
        threshold: The naive drop threshold.

    Returns:
        ``{metric: would this naive gate fail?}``.
    """
    return {item.metric: item.delta.value < -threshold for item in comparison.metrics}


def policy_fingerprint(policy: GatePolicy) -> str:
    """Short hash of the policy, printed in the verdict so loosened margins are visible."""
    return stable_hash(policy.model_dump(mode="json"), length=10)
