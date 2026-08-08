"""Gate engine (L6): non-inferiority verdicts, FDR control, tripwires, and honest uncertainty."""

from agentgate.gate.engine import (
    EXIT_ERROR,
    EXIT_PASS,
    EXIT_REGRESSION,
    EXIT_SAFETY,
    EXIT_UNDERPOWERED,
    GateResult,
    MetricRuling,
    evaluate,
    judge_blocked,
    naive_verdict,
    new_safety_failures,
    policy_fingerprint,
    resolve_margin,
)
from agentgate.gate.pipeline import (
    RELIABILITY_BASE,
    RELIABILITY_METRIC,
    build_comparison,
    collect_safety_findings,
    derive_pass_hat_k_results,
)

__all__ = [
    "EXIT_ERROR",
    "EXIT_PASS",
    "EXIT_REGRESSION",
    "EXIT_SAFETY",
    "EXIT_UNDERPOWERED",
    "RELIABILITY_BASE",
    "RELIABILITY_METRIC",
    "GateResult",
    "MetricRuling",
    "build_comparison",
    "collect_safety_findings",
    "derive_pass_hat_k_results",
    "evaluate",
    "judge_blocked",
    "naive_verdict",
    "new_safety_failures",
    "policy_fingerprint",
    "resolve_margin",
]
