"""The living harness: continuous recording, coverage tracking, leaderboards, and trends.

AgentGate is built to keep running. The gate answers one question about one pull request; this
package answers the standing question — what do we currently know about which models, how sure are
we, and what is the most valuable thing to measure next.

Everything here is derived from the run store, so there is no second source of truth to drift.
"""

from agentgate.harness.export import (
    DEFAULT_EXPORT,
    CellSnapshot,
    MetricSnapshot,
    build_snapshot,
    write_snapshot,
)
from agentgate.harness.leaderboard import (
    DEFAULT_METRIC,
    Leaderboard,
    Standing,
    assign_tiers,
    build_leaderboard,
    head_to_head,
)
from agentgate.harness.ledger import Cell, Ledger, LedgerEntry, model_of
from agentgate.harness.loop import CellOutcome, SessionReport, run_session
from agentgate.harness.schedule import Plan, PlannedCell, estimate_seconds, plan, suite_sizes
from agentgate.harness.trend import (
    IMPROVED,
    INCOMPARABLE,
    REGRESSED,
    STABLE,
    Movement,
    Trend,
    TrendPoint,
    build_trend,
)

__all__ = [
    "DEFAULT_EXPORT",
    "DEFAULT_METRIC",
    "IMPROVED",
    "INCOMPARABLE",
    "REGRESSED",
    "STABLE",
    "Cell",
    "CellOutcome",
    "CellSnapshot",
    "Leaderboard",
    "Ledger",
    "LedgerEntry",
    "MetricSnapshot",
    "Movement",
    "Plan",
    "PlannedCell",
    "SessionReport",
    "Standing",
    "Trend",
    "TrendPoint",
    "assign_tiers",
    "build_leaderboard",
    "build_snapshot",
    "build_trend",
    "estimate_seconds",
    "head_to_head",
    "model_of",
    "plan",
    "run_session",
    "suite_sizes",
    "write_snapshot",
]
