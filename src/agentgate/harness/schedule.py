"""Deciding what to measure next when there is never enough time to measure everything.

The harness runs against free tiers and a laptop, so its budget is measured in minutes and its
appetite is unbounded: every model times every suite times every K. The scheduler turns that into
an ordered queue under a wall-clock budget.

The ordering rule is **breadth before depth**, and it is a statistical claim rather than a
preference. A model with no recording at all carries unbounded uncertainty — nothing can be said
about it, and no comparison involving it is possible. A model with a partial recording carries
merely wide uncertainty. Spending the budget on the first kind therefore removes strictly more
uncertainty per minute than deepening the second, until every cell has been touched once.

Cost estimates come from measured throughput in the model catalogue. Where a model has never been
timed, the estimate is unknown, and unknown-cost cells are scheduled **last** — an unmeasured
model could be `qwen3:4b` at 75 seconds a call, and letting one of those in first can consume an
entire session's budget on a single cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agentgate.harness.ledger import Cell, Ledger
from agentgate.providers.models import ModelCard, get_card
from agentgate.runner.loader import discover_suites, load_suite

CALLS_PER_TASK = 4
"""Typical model round-trips per task. Measured on the smoke and tau2 suites."""

UNKNOWN_COST = float("inf")


@dataclass(frozen=True, slots=True)
class PlannedCell:
    """One cell the harness intends to record, with why and how expensive.

    Args:
        cell: What to record.
        n_tasks: Tasks in the suite, so cost is explainable rather than magic.
        est_seconds: Wall-clock estimate, or infinity when the model has never been timed.
        reason: Why this cell was queued, in plain words.
        resumable: True when a partial recording exists, so most units are already cached.
    """

    cell: Cell
    n_tasks: int
    est_seconds: float
    reason: str
    resumable: bool = False

    @property
    def cost_known(self) -> bool:
        """False when the model has no measured throughput."""
        return self.est_seconds != UNKNOWN_COST

    def describe_cost(self) -> str:
        """Human-readable cost estimate."""
        if not self.cost_known:
            return "unmeasured throughput"
        if self.est_seconds < 90:
            return f"~{self.est_seconds:.0f}s"
        if self.est_seconds < 5400:
            return f"~{self.est_seconds / 60:.0f} min"
        return f"~{self.est_seconds / 3600:.1f} h"


@dataclass(frozen=True, slots=True)
class Plan:
    """An ordered queue of work that fits a budget, plus what did not fit."""

    cells: tuple[PlannedCell, ...]
    deferred: tuple[PlannedCell, ...] = ()
    budget_seconds: float = 0.0

    @property
    def est_seconds(self) -> float:
        """Total estimated cost of the queued work."""
        return sum(planned.est_seconds for planned in self.cells if planned.cost_known)

    def describe(self) -> str:
        """One line summarising the plan."""
        if not self.cells:
            return "Nothing to record: every cell is complete, or nothing fits the budget."
        minutes = self.est_seconds / 60
        tail = f", {len(self.deferred)} deferred" if self.deferred else ""
        return f"{len(self.cells)} cells queued, ~{minutes:.0f} min estimated{tail}."


def estimate_seconds(card: ModelCard | None, *, n_tasks: int, k: int) -> float:
    """Estimate wall-clock cost of recording one cell.

    Returns infinity when the model has no measured rate — the scheduler treats that as a risk
    to be deferred rather than a number to be invented.
    """
    if card is None or card.approx_s_per_call is None:
        return UNKNOWN_COST
    return n_tasks * k * CALLS_PER_TASK * card.approx_s_per_call


def suite_sizes(root: str | Path = "suites") -> dict[str, int]:
    """Return ``{suite name: task count}`` for every discoverable suite."""
    sizes: dict[str, int] = {}
    for name, path in discover_suites(root).items():
        try:
            sizes[name] = len(load_suite(path).tasks)
        except Exception:  # a broken suite must not stop the harness from planning the rest
            continue
    return sizes


def plan(
    ledger: Ledger,
    *,
    suites: list[str],
    models: list[str],
    k: int,
    budget_seconds: float,
    sizes: dict[str, int] | None = None,
) -> Plan:
    """Order the outstanding work, newest evidence first, within a budget.

    Args:
        ledger: What has already been recorded.
        suites: Suite names to cover.
        models: Model ids to cover.
        k: Repetitions per task.
        budget_seconds: Wall-clock budget. Zero means unlimited.
        sizes: ``{suite: n_tasks}``; discovered from disk when omitted.

    Returns:
        The plan, with anything that did not fit reported as deferred rather than dropped.
    """
    sizes = sizes if sizes is not None else suite_sizes()
    candidates: list[PlannedCell] = []

    for cell in ledger.missing(suites=suites, models=models, k=k):
        n_tasks = sizes.get(cell.suite, 0)
        if n_tasks == 0:
            continue
        entry = ledger.latest(cell)
        card = get_card(cell.model_id)
        full = estimate_seconds(card, n_tasks=n_tasks, k=k)

        if entry is None:
            reason = "never measured"
            est = full
        elif not entry.is_scored:
            reason = "recorded but never scored"
            est = 0.0
        else:
            remaining = max(entry.expected_samples - entry.n_samples, 0)
            share = remaining / entry.expected_samples if entry.expected_samples else 1.0
            reason = f"partial: {entry.n_samples}/{entry.expected_samples} units recorded"
            est = full * share if full != UNKNOWN_COST else UNKNOWN_COST

        candidates.append(
            PlannedCell(
                cell=cell,
                n_tasks=n_tasks,
                est_seconds=est,
                reason=reason,
                resumable=entry is not None,
            )
        )

    ordered = sorted(candidates, key=_priority)
    if budget_seconds <= 0:
        return Plan(cells=tuple(ordered), budget_seconds=budget_seconds)

    queued: list[PlannedCell] = []
    deferred: list[PlannedCell] = []
    spent = 0.0
    for planned in ordered:
        cost = planned.est_seconds
        if not planned.cost_known or spent + cost > budget_seconds:
            deferred.append(planned)
            continue
        queued.append(planned)
        spent += cost
    return Plan(cells=tuple(queued), deferred=tuple(deferred), budget_seconds=budget_seconds)


def _priority(planned: PlannedCell) -> tuple[int, float, str]:
    """Sort key: breadth first, then cheapest, then stable by name.

    Rank 0 — already recorded, only scoring is missing. Nearly free, unlocks a leaderboard row.
    Rank 1 — never measured. Removes unbounded uncertainty.
    Rank 2 — partial. Narrows uncertainty that already exists.
    Rank 3 — cost unknown. Deferred so one slow model cannot eat a whole session.
    """
    if not planned.cost_known:
        rank = 3
    elif planned.est_seconds == 0.0:
        rank = 0
    elif not planned.resumable:
        rank = 1
    else:
        rank = 2
    return (rank, planned.est_seconds, str(planned.cell))
