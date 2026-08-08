"""Assemble a :class:`ComparisonResult` from two scored runs.

Three jobs the raw statistics layer does not do:

* **Derive reliability as a gateable metric.** ``reliability.pass_hat_k`` has no per-sample
  score — it is a function of a task's K outcomes. So it is synthesised into one per-task
  pseudo-sample, which then flows through exactly the same paired machinery as everything else.
  A separate code path for the headline reliability number would be the obvious place for a bug
  to hide.
* **Collect safety findings.** Per (task, repetition), so the report can name the exact task that
  started leaking.
* **Refuse to compare runs that are not paired.** Enforced here, at the boundary, rather than
  trusted upstream.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from agentgate.runner.pairing import assert_comparable
from agentgate.schemas.common import MetricFamily, stable_hash
from agentgate.schemas.policy import GatePolicy
from agentgate.schemas.results import (
    ComparisonResult,
    JudgeHealth,
    MetricResult,
    RunManifest,
    SafetyFinding,
)
from agentgate.stats.aggregate import repetition_scores, task_means
from agentgate.stats.compare import ComparisonConfig, compare_all
from agentgate.stats.reliability import (
    SUCCESS_THRESHOLD,
    outcomes_from_scores,
    pass_hat_k,
    reliability_report,
)

RELIABILITY_METRIC = "reliability.pass_hat_k"
RELIABILITY_BASE = "outcome.task_success"


def derive_pass_hat_k_results(
    results: Sequence[MetricResult], *, k: int, system: str
) -> list[MetricResult]:
    """Synthesise per-task ``pass^k`` pseudo-samples from a binary metric's repetitions.

    Args:
        results: Per-sample results for the binary base metric.
        k: The ``k`` the policy gates on.
        system: System label to record.

    Returns:
        One result per task, or an empty list when no task ran at least ``k`` repetitions.
    """
    scores = repetition_scores(results)
    if not scores:
        return []
    outcomes = outcomes_from_scores(scores, threshold=SUCCESS_THRESHOLD)
    derived: list[MetricResult] = []
    for outcome in outcomes:
        if outcome.trials < k:
            continue
        derived.append(
            MetricResult(
                metric=RELIABILITY_METRIC,
                task_id=outcome.task_id,
                rep=0,
                system=system,
                value=pass_hat_k(outcome.successes, outcome.trials, k),
                family=MetricFamily.RELIABILITY,
                dtype="proportion",
                direction="higher_is_better",
                detail={"successes": outcome.successes, "trials": outcome.trials, "k": k},
            )
        )
    return derived


def collect_safety_findings(
    baseline: Sequence[MetricResult],
    candidate: Sequence[MetricResult],
    tripwires: Sequence[str],
) -> list[SafetyFinding]:
    """Pair safety-metric samples and report every (task, repetition) where either side failed.

    Args:
        baseline: Baseline per-sample results.
        candidate: Candidate per-sample results.
        tripwires: Metrics to watch.

    Returns:
        Findings in (metric, task, rep) order. Findings where neither side failed are omitted;
        findings where only the baseline failed are kept, because a *fixed* tripwire is worth
        seeing too.
    """
    watched = set(tripwires)
    index: dict[tuple[str, str, int], list[MetricResult | None]] = {}
    for result in baseline:
        if result.metric in watched and result.is_scored:
            index.setdefault((result.metric, result.task_id, result.rep), [None, None])[0] = result
    for result in candidate:
        if result.metric in watched and result.is_scored:
            index.setdefault((result.metric, result.task_id, result.rep), [None, None])[1] = result

    findings: list[SafetyFinding] = []
    for (metric, task_id, rep), (base, cand) in sorted(index.items()):
        base_failed = bool(base and base.value)
        cand_failed = bool(cand and cand.value)
        if not base_failed and not cand_failed:
            continue
        findings.append(
            SafetyFinding(
                metric=metric,
                task_id=task_id,
                rep=rep,
                baseline_failed=base_failed,
                candidate_failed=cand_failed,
                detail=(cand.detail if cand else {}) or (base.detail if base else {}),
            )
        )
    return findings


def resolve_margins(
    policy: GatePolicy, baseline_results: Sequence[MetricResult]
) -> dict[str, float]:
    """Resolve every gated metric's margin to an absolute value, in metric units.

    Ratio margins (``margin_ratio: 1.25`` on latency) are relative allowances, so they need the
    baseline's scale before the paired tests can be shifted by them. Resolving them *here*, ahead
    of the comparison, is what keeps the hypothesis the tests answer identical to the one the
    gate rules on — computing the tests at margin 0 and applying the real margin afterwards would
    silently test the wrong thing.

    Args:
        policy: Gate policy.
        baseline_results: Baseline per-sample results, used for the baseline scale.

    Returns:
        ``{metric: absolute margin}``.
    """
    by_metric: dict[str, list[MetricResult]] = {}
    for result in baseline_results:
        by_metric.setdefault(result.metric, []).append(result)

    margins: dict[str, float] = {}
    for entry in policy.enabled_metrics:
        if entry.margin is not None:
            margins[entry.metric] = abs(entry.margin)
            continue
        ratio = entry.margin_ratio or 1.0
        means = task_means(repetition_scores(by_metric.get(entry.metric, [])))
        if not means:
            continue
        baseline_mean = abs(sum(means.values()) / len(means))
        margins[entry.metric] = abs(ratio - 1.0) * baseline_mean
    return margins


def build_comparison(
    *,
    baseline_manifest: RunManifest,
    candidate_manifest: RunManifest,
    baseline_results: Sequence[MetricResult],
    candidate_results: Sequence[MetricResult],
    policy: GatePolicy,
    clusters: dict[str, str] | None = None,
    judge_health: JudgeHealth | None = None,
    seed: int = 20260101,
    bootstrap_resamples: int = 2_000,
) -> ComparisonResult:
    """Build the full paired comparison the gate rules on.

    Args:
        baseline_manifest: Baseline run manifest.
        candidate_manifest: Candidate run manifest.
        baseline_results: Baseline per-sample metric results.
        candidate_results: Candidate per-sample metric results.
        policy: Gate policy, which supplies margins and levels.
        clusters: ``{task_id: cluster_id}``.
        judge_health: Judge panel to attach.
        seed: RNG seed for permutation and bootstrap.
        bootstrap_resamples: Replicates for the reliability panels.

    Returns:
        The comparison.

    Raises:
        SuiteMismatchError: When the two runs are not paired on identical inputs (C2).
    """
    assert_comparable(baseline_manifest, candidate_manifest)

    base_list = list(baseline_results)
    cand_list = list(candidate_results)

    reliability_entry = next(
        (item for item in policy.enabled_metrics if item.metric == RELIABILITY_METRIC), None
    )
    if reliability_entry is not None:
        k = reliability_entry.k or baseline_manifest.k
        base_source = [item for item in base_list if item.metric == RELIABILITY_BASE]
        cand_source = [item for item in cand_list if item.metric == RELIABILITY_BASE]
        base_list.extend(derive_pass_hat_k_results(base_source, k=k, system="baseline"))
        cand_list.extend(derive_pass_hat_k_results(cand_source, k=k, system="candidate"))

    margins = resolve_margins(policy, base_list)
    config = ComparisonConfig(
        alpha=policy.alpha,
        normality_alpha=policy.normality_alpha,
        permutation_iters=policy.permutation_iters,
        power_target=policy.power_target,
        use_clusters=policy.use_clustered_se,
        seed=seed,
        margins=margins,
    )
    comparisons = compare_all(base_list, cand_list, clusters=clusters, config=config)

    reliability_base = [item for item in base_list if item.metric == RELIABILITY_BASE]
    reliability_cand = [item for item in cand_list if item.metric == RELIABILITY_BASE]
    reliability_reports_baseline = (
        [
            reliability_report(
                RELIABILITY_BASE,
                repetition_scores(reliability_base),
                clusters=clusters,
                resamples=bootstrap_resamples,
                seed=seed,
            )
        ]
        if reliability_base
        else []
    )
    reliability_reports_candidate = (
        [
            reliability_report(
                RELIABILITY_BASE,
                repetition_scores(reliability_cand),
                clusters=clusters,
                resamples=bootstrap_resamples,
                seed=seed,
            )
        ]
        if reliability_cand
        else []
    )

    findings = collect_safety_findings(base_list, cand_list, policy.safety_tripwires)
    n_clusters = len(set((clusters or {}).values())) if clusters else 0
    warnings: list[str] = []
    if candidate_manifest.git_dirty or baseline_manifest.git_dirty:
        warnings.append(
            "one or both runs were produced from a dirty working tree; the recorded git SHA "
            "does not fully describe what ran"
        )

    return ComparisonResult(
        comparison_id=stable_hash(
            [baseline_manifest.run_id, candidate_manifest.run_id, policy.policy_hash()], length=12
        ),
        created_at=datetime.now(UTC),
        baseline_run_id=baseline_manifest.run_id,
        candidate_run_id=candidate_manifest.run_id,
        suite=candidate_manifest.suite,
        k=candidate_manifest.k,
        clustered=bool(clusters) and n_clusters < candidate_manifest.suite.n_tasks,
        n_clusters=n_clusters,
        metrics=comparisons,
        reliability_baseline=reliability_reports_baseline,
        reliability_candidate=reliability_reports_candidate,
        safety_findings=findings,
        judge_health=judge_health,
        warnings=warnings,
    )
