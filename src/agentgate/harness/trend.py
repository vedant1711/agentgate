"""How a model's score on a suite has moved over time.

A running project accumulates repeated measurements of the same cell, and the obvious thing to do
with them — draw a line and read a slope — is also the easiest way to manufacture a finding. Two
points differing by 3 points mean nothing when each carries a 12-point interval.

So a trend here reports **movement only when consecutive intervals fail to overlap**, and labels
everything else as noise. It also refuses to compare across a changed suite: if the suite's
content hash moved between two recordings, the scores answer different questions and the trend
says so rather than drawing a line through the discontinuity.

This is the same rule the gate applies to a PR, turned on the harness itself.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from agentgate.harness.ledger import Cell, Ledger
from agentgate.schemas.results import Estimate
from agentgate.stats.aggregate import summarise_metric

if TYPE_CHECKING:
    from agentgate.storage.duckdb_store import RunStore

STABLE = "stable"
IMPROVED = "improved"
REGRESSED = "regressed"
INCOMPARABLE = "incomparable"


@dataclass(frozen=True, slots=True)
class TrendPoint:
    """One measurement in a cell's history."""

    run_id: str
    recorded_at: datetime
    estimate: Estimate
    suite_hash: str
    git_sha: str
    n_tasks: int

    @property
    def interval(self) -> tuple[float, float] | None:
        """The confidence interval, or ``None`` for a degenerate sample."""
        if self.estimate.ci_low is None or self.estimate.ci_high is None:
            return None
        return (self.estimate.ci_low, self.estimate.ci_high)


@dataclass(frozen=True, slots=True)
class Movement:
    """The relationship between two consecutive measurements."""

    earlier: TrendPoint
    later: TrendPoint
    verdict: str
    delta: float

    def describe(self) -> str:
        """One line explaining the movement and why it was called that way."""
        if self.verdict == INCOMPARABLE:
            return "suite changed between these runs; the scores answer different questions"
        if self.verdict == STABLE:
            return f"moved {self.delta:+.3f}, but the intervals overlap — not established"
        return f"{self.verdict} by {abs(self.delta):.3f}; intervals do not overlap"


@dataclass(frozen=True, slots=True)
class Trend:
    """A cell's measurement history and the movements between consecutive points."""

    cell: Cell
    metric: str
    direction: str
    points: tuple[TrendPoint, ...]
    movements: tuple[Movement, ...]

    @property
    def established_moves(self) -> list[Movement]:
        """Only the movements this evidence actually supports."""
        return [move for move in self.movements if move.verdict in (IMPROVED, REGRESSED)]

    def describe(self) -> str:
        """An honest summary of the history."""
        if len(self.points) < 2:
            return f"{len(self.points)} measurement(s); a trend needs at least two."
        established = self.established_moves
        if not established:
            return (
                f"{len(self.points)} measurements, no established movement in {self.metric}: "
                f"every consecutive pair has overlapping intervals."
            )
        latest = established[-1]
        return f"{len(established)} established movement(s); most recent: {latest.describe()}"


def build_trend(
    store: RunStore,
    *,
    cell: Cell,
    metric: str,
    ledger: Ledger | None = None,
    level: float = 0.95,
) -> Trend:
    """Assemble a cell's history for one metric, oldest first.

    Args:
        store: The run database.
        cell: The model/suite/K combination to trace.
        metric: Metric to trace.
        ledger: Prebuilt ledger.
        level: Confidence level for each point's interval.

    Returns:
        The trend. Points that were never scored for this metric are omitted, not zero-filled.
    """
    ledger = ledger or Ledger.from_store(store)
    points: list[TrendPoint] = []
    direction = "higher_is_better"

    for entry in reversed(ledger.history(cell)):
        manifest = store.get_run(entry.run_id)
        if manifest is None:
            continue
        summary = summarise_metric(
            store.load_scores(entry.run_id, metric=metric),
            clusters=store.clusters_for(entry.run_id),
            level=level,
        )
        if summary is None:
            continue
        direction = summary.direction
        points.append(
            TrendPoint(
                run_id=entry.run_id,
                recorded_at=entry.recorded_at,
                estimate=summary.clustered or summary.estimate,
                suite_hash=manifest.suite.content_hash,
                git_sha=entry.git_sha,
                n_tasks=summary.n_tasks,
            )
        )

    movements = tuple(
        _movement(earlier, later, direction=direction)
        for earlier, later in itertools.pairwise(points)
    )
    return Trend(
        cell=cell,
        metric=metric,
        direction=direction,
        points=tuple(points),
        movements=movements,
    )


def _movement(earlier: TrendPoint, later: TrendPoint, *, direction: str) -> Movement:
    """Classify the change between two consecutive measurements."""
    delta = later.estimate.value - earlier.estimate.value
    if earlier.suite_hash != later.suite_hash:
        return Movement(earlier, later, INCOMPARABLE, delta)

    left, right = earlier.interval, later.interval
    if left is None or right is None or (left[0] <= right[1] and right[0] <= left[1]):
        return Movement(earlier, later, STABLE, delta)

    better = delta > 0 if direction != "lower_is_better" else delta < 0
    return Movement(earlier, later, IMPROVED if better else REGRESSED, delta)
