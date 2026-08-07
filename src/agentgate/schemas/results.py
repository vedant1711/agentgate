"""Result schemas: per-sample scores, run manifests, and statistical summaries.

Design principle from C: *no number appears in any report without its interval*. That rule is
enforced structurally here — aggregate values are carried by :class:`Estimate`, which cannot be
constructed without stating the method used to derive its uncertainty.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, computed_field

from agentgate.schemas.common import (
    SCHEMA_VERSION,
    AgentGateModel,
    Direction,
    DType,
    FrozenModel,
    MetricFamily,
    ProviderMode,
    Verdict,
    stable_hash,
)
from agentgate.schemas.task import SuiteRef

# ---------------------------------------------------------------------------
# Per-sample scoring
# ---------------------------------------------------------------------------

SampleStatus = Literal["ok", "skipped", "error"]


class MetricResult(AgentGateModel):
    """One metric's score for one (task, repetition) sample.

    ``value`` is ``None`` whenever ``status`` is not ``"ok"``. Skipped samples are excluded
    from aggregation rather than treated as zeros — scoring a missing reference as 0 would
    silently bias the suite mean downward.
    """

    metric: str
    task_id: str
    rep: int = Field(ge=0)
    system: str
    value: float | None = None
    family: MetricFamily
    dtype: DType
    direction: Direction = "higher_is_better"
    status: SampleStatus = "ok"
    error: str | None = None
    detail: dict[str, Any] = Field(
        default_factory=dict, description="Audit trail: what matched, what didn't, and why."
    )
    judge_samples: list[float] = Field(
        default_factory=list, description="J judge draws for this item (C5)."
    )
    judge_variance: float | None = Field(
        default=None, description="Within-item judge variance, folded into the metric SE (C5)."
    )
    cost_tokens: int = Field(default=0, ge=0, description="Tokens spent scoring this sample.")

    @property
    def is_scored(self) -> bool:
        """True when this sample contributes to aggregation."""
        return self.status == "ok" and self.value is not None


# ---------------------------------------------------------------------------
# Run identity and reproducibility
# ---------------------------------------------------------------------------


class ModelRef(FrozenModel):
    """A pinned model used in a role. Version pinning is what makes judge drift detectable."""

    role: Literal["agent", "judge", "embedding"]
    model_id: str
    provider: str = ""
    version: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class BudgetSpec(FrozenModel):
    """Hard caps enforced by the runner (A3.6). Zero means unlimited."""

    max_requests: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=0, ge=0)
    max_cost_usd: float = Field(default=0.0, ge=0.0)
    max_wall_s: float = Field(default=0.0, ge=0.0)


class RunManifest(FrozenModel):
    """Everything needed to reproduce a run byte-for-byte (A3.4).

    :attr:`config_hash` deliberately excludes wall-clock and host details so that two runs on
    different machines with the same inputs and the same replay cache hash identically.
    """

    schema_version: int = SCHEMA_VERSION
    run_id: str
    created_at: datetime
    agentgate_version: str
    git_sha: str = "unknown"
    git_dirty: bool = False

    suite: SuiteRef
    system: str
    agent: str
    k: int = Field(ge=1)
    base_seed: int
    mode: ProviderMode
    models: list[ModelRef] = Field(default_factory=list)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    library_versions: dict[str, str] = Field(default_factory=dict)
    faults: list[str] = Field(default_factory=list)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    host: dict[str, str] = Field(
        default_factory=dict, description="Platform info. Excluded from config_hash by design."
    )
    notes: str = ""

    def reproducibility_key(self) -> dict[str, Any]:
        """Return the subset of the manifest that determines run output."""
        return {
            "schema_version": self.schema_version,
            "suite": self.suite.model_dump(mode="json"),
            "agent": self.agent,
            "k": self.k,
            "base_seed": self.base_seed,
            "mode": self.mode.value,
            "models": [m.model_dump(mode="json") for m in self.models],
            "prompt_hashes": self.prompt_hashes,
            "faults": sorted(self.faults),
        }

    @computed_field  # type: ignore[prop-decorator]
    @property
    def config_hash(self) -> str:
        """Deterministic hash of :meth:`reproducibility_key`."""
        return stable_hash(self.reproducibility_key())

    def is_comparable_to(self, other: RunManifest) -> tuple[bool, str]:
        """Check whether two runs may be paired (C2).

        Comparisons across mismatched suites, K, or seeds are *refused*, not warned about,
        because pairing per-task differences requires identical task instances.

        Args:
            other: The other run's manifest.

        Returns:
            ``(ok, reason)`` — ``reason`` is empty when ``ok`` is True.
        """
        if self.suite.content_hash != other.suite.content_hash:
            return False, (
                f"suite content hash differs: {self.suite.name}@{self.suite.content_hash} "
                f"vs {other.suite.name}@{other.suite.content_hash}"
            )
        if self.k != other.k:
            return False, f"K differs: {self.k} vs {other.k}"
        if self.base_seed != other.base_seed:
            return False, f"base seed differs: {self.base_seed} vs {other.base_seed}"
        if self.schema_version != other.schema_version:
            return False, f"schema version differs: {self.schema_version} vs {other.schema_version}"
        return True, ""


class RunSummary(FrozenModel):
    """Aggregate bookkeeping for a completed run."""

    run_id: str
    manifest: RunManifest
    n_tasks: int
    n_samples: int
    status_counts: dict[str, int] = Field(default_factory=dict)
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    wall_seconds: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


# ---------------------------------------------------------------------------
# Statistical containers
# ---------------------------------------------------------------------------


class Estimate(FrozenModel):
    """A point estimate that cannot exist without its uncertainty.

    ``se`` and the interval may be ``None`` only for degenerate samples (n < 2), in which case
    ``method`` records why.
    """

    value: float
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    ci_level: float = Field(default=0.95, gt=0, lt=1)
    method: str = Field(description="Which interval procedure produced this, e.g. 'wilson'.")
    n: int = Field(ge=0, description="Independent units (tasks, or clusters when clustered).")

    @property
    def half_width(self) -> float | None:
        """Half the CI width, or ``None`` when no interval was computable."""
        if self.ci_low is None or self.ci_high is None:
            return None
        return (self.ci_high - self.ci_low) / 2.0


class VarianceDecomposition(FrozenModel):
    """Between-task vs within-task variance split (C1.5).

    The intraclass correlation answers the free-tier budgeting question directly: high ICC
    means buy more tasks, low ICC means buy more repetitions.
    """

    between_task_var: float = Field(ge=0.0)
    within_task_var: float = Field(ge=0.0)
    icc: float = Field(description="Between / (between + within); may be clipped to 0.")
    n_tasks: int = Field(ge=0)
    k: int = Field(ge=1)
    recommendation: str = ""


class MetricSummary(FrozenModel):
    """Single-run inference for one metric over one suite (C1)."""

    metric: str
    family: MetricFamily
    dtype: DType
    direction: Direction
    n_tasks: int = Field(ge=0)
    n_scored_samples: int = Field(ge=0)
    n_skipped: int = Field(default=0, ge=0)
    k: int = Field(ge=1)
    estimate: Estimate
    clustered: Estimate | None = Field(
        default=None, description="Cluster-robust version; present only when clusters exist."
    )
    se_inflation: float | None = Field(
        default=None, description="clustered_se / naive_se — the number E3 says can exceed 3."
    )
    variance: VarianceDecomposition | None = None
    per_task: dict[str, float] = Field(
        default_factory=dict, description="Task-level means over K reps (the analysis unit)."
    )
    judge_backed: bool = False


class TaskFlake(FrozenModel):
    """A task that succeeded only sometimes — the CI-killing behaviour (B6)."""

    task_id: str
    successes: int = Field(ge=0)
    k: int = Field(ge=1)

    @property
    def rate(self) -> float:
        """Empirical success rate across repetitions."""
        return self.successes / self.k


class PassKPoint(FrozenModel):
    """pass@k and pass^k at one value of k (B6, E2)."""

    k: int = Field(ge=1)
    pass_at_k: Estimate
    pass_hat_k: Estimate


class ReliabilityReport(FrozenModel):
    """The reliability family for one binary base metric."""

    base_metric: str
    k_max: int = Field(ge=1)
    curve: list[PassKPoint] = Field(default_factory=list)
    flake_rate: Estimate
    score_variance: float = 0.0
    flakiest_tasks: list[TaskFlake] = Field(default_factory=list)


class RunReport(FrozenModel):
    """Full single-run analysis: every metric with uncertainty, plus reliability."""

    run_id: str
    manifest: RunManifest
    summaries: list[MetricSummary] = Field(default_factory=list)
    reliability: list[ReliabilityReport] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def summary(self, metric: str) -> MetricSummary | None:
        """Return the summary for ``metric``, or ``None`` if it was not scored."""
        return next((s for s in self.summaries if s.metric == metric), None)


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------


class PairedTestResult(FrozenModel):
    """Outcome of one hypothesis test on paired differences (C2)."""

    test: str = Field(description="mcnemar_exact | paired_t | wilcoxon | permutation")
    statistic: float | None = None
    p_one_sided: float = Field(ge=0.0, le=1.0)
    p_two_sided: float | None = Field(default=None, ge=0.0, le=1.0)
    df: float | None = None
    n_pairs: int = Field(ge=0)
    detail: dict[str, float] = Field(default_factory=dict)
    selection_reason: str = Field(
        default="", description="Why this test was chosen (e.g. Shapiro-Wilk p on d_i)."
    )


class PowerReport(FrozenModel):
    """Power and minimum detectable effect for the paired design (C4)."""

    sigma_d: float = Field(ge=0.0)
    n_pairs: int = Field(ge=0)
    alpha: float = Field(gt=0, lt=1)
    power_target: float = Field(gt=0, lt=1)
    achieved_power: float = Field(ge=0.0, le=1.0, description="Power to detect the margin at n.")
    mde: float = Field(ge=0.0, description="Smallest detectable effect at target power and n.")
    n_required: int = Field(ge=0, description="Pairs needed to detect the configured margin.")
    method: str = ""


class EffectSize(FrozenModel):
    """Standardised effect size with its kind named, since kinds are not interchangeable."""

    kind: Literal["cohens_dz", "odds_ratio", "risk_difference", "cliffs_delta"]
    value: float
    interpretation: str = ""


class MetricComparison(FrozenModel):
    """Everything the gate needs to rule on one metric (C2 + C3)."""

    metric: str
    family: MetricFamily
    dtype: DType
    direction: Direction
    n_pairs: int = Field(ge=0)
    baseline: Estimate
    candidate: Estimate
    delta: Estimate = Field(
        description="Direction-normalised mean difference (candidate - baseline) with CI."
    )
    margin: float = Field(description="Non-inferiority margin delta_m, always non-negative.")
    margin_kind: Literal["absolute", "ratio"] = "absolute"
    correlation: float | None = Field(
        default=None, description="Pearson r between paired scores — the variance pairing bought."
    )
    effect_size: EffectSize | None = None
    regression_test: PairedTestResult
    noninferiority_test: PairedTestResult
    permutation_test: PairedTestResult | None = None
    p_adj: float | None = Field(default=None, description="Benjamini-Hochberg adjusted p (C3).")
    power: PowerReport | None = None
    clustered: bool = False
    verdict: Verdict = Verdict.INCONCLUSIVE
    notes: list[str] = Field(default_factory=list)


class SafetyFinding(FrozenModel):
    """A safety tripwire hit. Bypasses statistics entirely (C3)."""

    metric: str
    task_id: str
    rep: int
    baseline_failed: bool
    candidate_failed: bool
    detail: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_new(self) -> bool:
        """True when the candidate introduced this failure."""
        return self.candidate_failed and not self.baseline_failed


class JudgeHealth(FrozenModel):
    """Judge trustworthiness panel published next to every judge-backed number (D3, D4)."""

    judge_model: str = ""
    cohens_kappa: float | None = None
    spearman_rho: float | None = None
    n_calibration_items: int = 0
    position_flip_rate: float | None = None
    verbosity_correlation: float | None = None
    markdown_correlation: float | None = None
    drift_detected: bool = False
    drift_detail: str = ""
    self_judging: bool = False
    warnings: list[str] = Field(default_factory=list)

    @property
    def meets_gate_bar(self) -> bool:
        """True when kappa clears the D4 bar of 0.6 required to back a gated metric."""
        return self.cohens_kappa is not None and self.cohens_kappa >= 0.6


class ComparisonResult(FrozenModel):
    """The paired analysis of two runs, before the gate renders its verdict."""

    schema_version: int = SCHEMA_VERSION
    comparison_id: str
    created_at: datetime
    baseline_run_id: str
    candidate_run_id: str
    suite: SuiteRef
    k: int = Field(ge=1)
    clustered: bool = False
    n_clusters: int = 0
    metrics: list[MetricComparison] = Field(default_factory=list)
    reliability_baseline: list[ReliabilityReport] = Field(default_factory=list)
    reliability_candidate: list[ReliabilityReport] = Field(default_factory=list)
    safety_findings: list[SafetyFinding] = Field(default_factory=list)
    judge_health: JudgeHealth | None = None
    warnings: list[str] = Field(default_factory=list)

    def comparison(self, metric: str) -> MetricComparison | None:
        """Return the comparison for ``metric``, or ``None``."""
        return next((c for c in self.metrics if c.metric == metric), None)

    @property
    def new_safety_failures(self) -> list[SafetyFinding]:
        """Safety failures the candidate introduced."""
        return [f for f in self.safety_findings if f.is_new]


class GateVerdict(FrozenModel):
    """The machine-readable gate decision written to ``gate.json`` (L6)."""

    schema_version: int = SCHEMA_VERSION
    comparison_id: str
    created_at: datetime
    verdict: Verdict
    exit_code: int = Field(ge=0, le=125)
    baseline_run_id: str
    candidate_run_id: str
    suite: SuiteRef
    metric_verdicts: dict[str, Verdict] = Field(default_factory=dict)
    failing_metrics: list[str] = Field(default_factory=list)
    underpowered_metrics: list[str] = Field(default_factory=list)
    safety_failures: list[SafetyFinding] = Field(default_factory=list)
    summary: str = ""
    warnings: list[str] = Field(default_factory=list)
    policy_hash: str = ""

    @property
    def passed(self) -> bool:
        """True when the gate lets the change through."""
        return self.exit_code == 0
