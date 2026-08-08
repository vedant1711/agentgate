"""The continuous recorder: the part that makes this a running project rather than a report.

One session of the harness takes a plan, records cells until a wall-clock deadline, scores what it
recorded, and stops. Everything it produces lands in the same store and the same provider cache,
so the next session resumes where this one stopped and the evidence base only grows.

Three properties matter more than throughput here, because this loop is meant to run unattended
against flaky free tiers on a laptop that sleeps:

* **A failing cell must not end the session.** A dead provider, an expired key, a model that
  refuses tool schemas — each fails that cell, is recorded as a failure with its reason, and the
  loop moves to the next. Anything else means one bad key silently costs a night of recording.
* **The deadline is checked between cells, never inside one.** A half-recorded cell is resumable;
  a cell abandoned mid-scoring is a partial write. The loop would rather overrun its budget by one
  cell than leave inconsistent state, and it says by how much it overran.
* **Recording and scoring are one step.** An unscored recording cannot enter a leaderboard, so
  splitting them just creates a backlog of work that looks done and is not.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from agentgate.harness.ledger import Cell
from agentgate.harness.schedule import Plan, PlannedCell
from agentgate.metrics import MetricsEngine
from agentgate.runner import RunConfig, Runner
from agentgate.runner.loader import discover_suites
from agentgate.schemas.common import ProviderMode
from agentgate.storage.duckdb_store import RunStore

Announce = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class CellOutcome:
    """What happened to one cell in a session."""

    cell: Cell
    ok: bool
    seconds: float
    run_id: str = ""
    n_samples: int = 0
    n_completed: int = 0
    n_scores: int = 0
    resumed: int = 0
    error: str = ""

    @property
    def status(self) -> str:
        """One-word status for tables and logs."""
        return "recorded" if self.ok else "failed"


@dataclass(slots=True)
class SessionReport:
    """Everything one harness session did."""

    started_at: datetime
    budget_seconds: float
    outcomes: list[CellOutcome] = field(default_factory=list)
    skipped: list[PlannedCell] = field(default_factory=list)
    """Cells the deadline cut before they were attempted."""

    @property
    def seconds(self) -> float:
        """Wall time actually spent recording."""
        return sum(outcome.seconds for outcome in self.outcomes)

    @property
    def recorded(self) -> list[CellOutcome]:
        """Cells that succeeded."""
        return [outcome for outcome in self.outcomes if outcome.ok]

    @property
    def failed(self) -> list[CellOutcome]:
        """Cells that failed, with their reasons."""
        return [outcome for outcome in self.outcomes if not outcome.ok]

    @property
    def overrun_seconds(self) -> float:
        """How far past the budget the session ran, finishing its last cell."""
        if self.budget_seconds <= 0:
            return 0.0
        return max(0.0, self.seconds - self.budget_seconds)

    def describe(self) -> str:
        """A short, honest summary of the session."""
        parts = [
            f"{len(self.recorded)} cells recorded",
            f"{len(self.failed)} failed",
            f"{self.seconds / 60:.1f} min",
        ]
        if self.skipped:
            parts.append(f"{len(self.skipped)} left for next session")
        if self.overrun_seconds > 0:
            parts.append(f"overran budget by {self.overrun_seconds / 60:.1f} min")
        return ", ".join(parts) + "."


def run_session(
    plan: Plan,
    *,
    store_path: Path,
    mode: ProviderMode = ProviderMode.CACHE,
    concurrency: int = 3,
    base_seed: int = 20260101,
    suite_root: str | Path = "suites",
    announce: Announce | None = None,
) -> SessionReport:
    """Record every cell in ``plan`` until its budget runs out.

    Args:
        plan: The ordered work queue.
        store_path: DuckDB file all runs are persisted into. Sharing one store is what lets the
            leaderboard compare models that were recorded weeks apart.
        mode: Provider mode. ``cache`` is the right default — it calls the model when a response
            is absent and reuses it forever after, which is exactly how the evidence base grows
            without re-paying for what is already known.
        concurrency: Parallel units per cell.
        base_seed: Base seed, held fixed across models so every model faces identical task
            instances. This is what makes the paired head-to-head comparison valid; varying it
            per model would silently destroy the pairing that gives the harness its power.
        suite_root: Where to find suites.
        announce: Optional progress callback, one line per event.

    Returns:
        The session report.
    """
    say = announce or (lambda _: None)
    suites = discover_suites(suite_root)
    report = SessionReport(started_at=datetime.now(UTC), budget_seconds=plan.budget_seconds)
    deadline = time.monotonic() + plan.budget_seconds if plan.budget_seconds > 0 else None

    for index, planned in enumerate(plan.cells, start=1):
        if deadline is not None and time.monotonic() >= deadline:
            report.skipped.extend(plan.cells[index - 1 :])
            say(f"budget exhausted; {len(report.skipped)} cells left for the next session")
            break

        cell = planned.cell
        suite_path = suites.get(cell.suite)
        if suite_path is None:
            report.outcomes.append(
                CellOutcome(
                    cell=cell, ok=False, seconds=0.0, error=f"suite {cell.suite!r} not found"
                )
            )
            continue

        say(f"[{index}/{len(plan.cells)}] {cell} — {planned.reason} ({planned.describe_cost()})")
        report.outcomes.append(
            _record_cell(
                cell,
                suite_path=suite_path,
                store_path=store_path,
                mode=mode,
                concurrency=concurrency,
                base_seed=base_seed,
            )
        )
        say(f"    {report.outcomes[-1].status}: {_describe(report.outcomes[-1])}")

    report.skipped.extend(plan.deferred)
    return report


def _record_cell(
    cell: Cell,
    *,
    suite_path: Path,
    store_path: Path,
    mode: ProviderMode,
    concurrency: int,
    base_seed: int,
) -> CellOutcome:
    """Record and score one cell, converting any failure into a reported outcome.

    The broad except is deliberate and load-bearing: this loop runs unattended against providers
    that fail in ways no exception hierarchy anticipates, and a single unhandled error would end
    a session that had hours of budget left.
    """
    import asyncio

    started = time.monotonic()
    try:
        config = RunConfig(
            suite_path=suite_path,
            system="baseline",
            k=cell.k,
            mode=mode,
            base_seed=base_seed,
            concurrency=concurrency,
            model=cell.model_id,
            store_path=store_path,
            resume=True,
        )
        runner = Runner(config)
        result = asyncio.run(runner.run())

        engine = MetricsEngine()
        scores = engine.score_run(runner.suite, result.trajectories, run_id=result.run_id)
        with RunStore(store_path) as store:
            store.save_scores(result.run_id, scores)

        completed = sum(1 for t in result.trajectories if t.status.value == "completed")
        return CellOutcome(
            cell=cell,
            ok=True,
            seconds=time.monotonic() - started,
            run_id=result.run_id,
            n_samples=len(result.trajectories),
            n_completed=completed,
            n_scores=sum(1 for score in scores if score.is_scored),
            resumed=result.resumed,
        )
    except Exception as exc:  # one bad cell must not end a session; see the docstring
        return CellOutcome(
            cell=cell,
            ok=False,
            seconds=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def _describe(outcome: CellOutcome) -> str:
    """Render a cell outcome as one readable line."""
    if not outcome.ok:
        return outcome.error
    resumed = f", {outcome.resumed} resumed" if outcome.resumed else ""
    return (
        f"{outcome.n_completed}/{outcome.n_samples} units completed{resumed}, "
        f"{outcome.n_scores} scores, {outcome.seconds:.0f}s"
    )
