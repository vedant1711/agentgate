"""Gate policy schema — the contents of ``gate.yaml`` (E3).

The policy is content-hashed and recorded in every verdict, so a reviewer can tell whether a
gate passed because the agent improved or because somebody loosened the margins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from agentgate.schemas.common import SCHEMA_VERSION, FrozenModel

UnderpoweredBehavior = Literal["warn", "fail"]


class GatedMetric(FrozenModel):
    """One metric the gate rules on, with its non-inferiority margin.

    Exactly one of ``margin`` (absolute, in metric units) or ``margin_ratio`` (multiplicative,
    for lower-is-better metrics like latency) must be set.
    """

    metric: str
    margin: float | None = Field(
        default=None,
        ge=0.0,
        description="Absolute non-inferiority margin delta_m in the metric's own units.",
    )
    margin_ratio: float | None = Field(
        default=None,
        gt=0.0,
        description="Multiplicative margin for scale metrics, e.g. 1.25 = 'up to 25% slower'.",
    )
    k: int | None = Field(
        default=None, ge=1, description="Which k to gate on for reliability.pass_hat_k."
    )
    enabled: bool = True
    note: str = ""

    @model_validator(mode="after")
    def _exactly_one_margin(self) -> Self:
        has_abs = self.margin is not None
        has_ratio = self.margin_ratio is not None
        if has_abs == has_ratio:
            msg = f"gated metric {self.metric!r}: set exactly one of margin / margin_ratio"
            raise ValueError(msg)
        return self

    @property
    def margin_kind(self) -> Literal["absolute", "ratio"]:
        """Which margin flavour this metric uses."""
        return "absolute" if self.margin is not None else "ratio"

    @property
    def resolved_name(self) -> str:
        """Metric key including the ``k`` suffix for reliability metrics."""
        if self.k is not None:
            return f"{self.metric}@{self.k}"
        return self.metric


class GatePolicy(FrozenModel):
    """Full gate configuration."""

    schema_version: int = SCHEMA_VERSION
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0, description="Per-test significance level.")
    fdr_q: float = Field(
        default=0.05, gt=0.0, lt=1.0, description="Benjamini-Hochberg FDR level across metrics."
    )
    power_target: float = Field(default=0.80, gt=0.0, lt=1.0)
    underpowered_behavior: UnderpoweredBehavior = Field(
        default="warn",
        description="'warn' passes an underpowered suite with a banner; 'strict' mode fails it.",
    )
    min_pairs: int = Field(
        default=5, ge=2, description="Below this many paired tasks, no verdict is claimed."
    )
    permutation_iters: int = Field(default=20_000, ge=1_000)
    bootstrap_iters: int = Field(default=10_000, ge=1_000)
    normality_alpha: float = Field(
        default=0.05,
        gt=0.0,
        lt=1.0,
        description="Shapiro-Wilk threshold for auto-selecting Wilcoxon over paired t (C2).",
    )
    use_clustered_se: bool = Field(
        default=True, description="Use cluster-robust SEs when the suite declares clusters."
    )
    require_judge_kappa: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Judge-human agreement floor before a judge may back a gated metric (D4).",
    )
    allow_self_judging: bool = Field(
        default=False, description="Override the judge/agent family independence rule (D3)."
    )
    gated_metrics: list[GatedMetric] = Field(default_factory=list)
    safety_tripwires: list[str] = Field(
        default_factory=list,
        description="Metrics that fail the gate on any new failure, bypassing statistics.",
    )

    @model_validator(mode="after")
    def _unique_metrics(self) -> Self:
        seen: set[str] = set()
        for entry in self.gated_metrics:
            if entry.resolved_name in seen:
                msg = f"duplicate gated metric {entry.resolved_name!r}"
                raise ValueError(msg)
            seen.add(entry.resolved_name)
        return self

    @property
    def enabled_metrics(self) -> list[GatedMetric]:
        """Gated metrics with ``enabled: true``."""
        return [entry for entry in self.gated_metrics if entry.enabled]

    def policy_hash(self) -> str:
        """Content hash recorded in the verdict for auditability."""
        return self.content_digest()

    @classmethod
    def load(cls, path: str | Path) -> GatePolicy:
        """Load and validate a ``gate.yaml`` file.

        Args:
            path: Path to the YAML policy.

        Returns:
            The validated policy.
        """
        return cls.model_validate_yaml(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def default(cls) -> GatePolicy:
        """The policy shipped in the repo root — a sensible starting point (E3)."""
        return cls(
            gated_metrics=[
                GatedMetric(metric="outcome.task_success", margin=0.03),
                GatedMetric(metric="reliability.pass_hat_k", k=4, margin=0.05),
                GatedMetric(metric="trajectory.f1", margin=0.03),
                GatedMetric(metric="rag.faithfulness", margin=0.05),
                GatedMetric(metric="efficiency.latency_ms", margin_ratio=1.25),
            ],
            safety_tripwires=[
                "safety.prompt_injection_compliance",
                "safety.pii_leak",
                "safety.forbidden_tool_invocation",
            ],
        )
