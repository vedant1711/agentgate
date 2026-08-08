"""What the harness has measured, and — more importantly — what it has not.

AgentGate is a *running* project: it accumulates evidence about models over time rather than
producing one report and stopping. That only works if the system can answer two questions at any
moment: what do we know, and what is the most valuable thing we do not know yet.

The ledger answers the first. It is a **derived view, not a second source of truth** — every entry
is reconstructed from the run store, so the ledger can never drift from the runs it describes and
there is no bookkeeping file to corrupt or forget to update.

The unit of work is a :class:`Cell`: one model, on one suite, at one K. Cells are deliberately
coarse. A finer unit would let the harness record half a suite against a model and report it
alongside complete measurements, and a leaderboard built from ragged coverage compares models on
different tasks while looking like it compares them on the same ones.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from agentgate.schemas.results import RunManifest
from agentgate.schemas.trajectory import RunStatus

if TYPE_CHECKING:
    from agentgate.storage.duckdb_store import RunStore

UNKNOWN_MODEL = "unknown"


@dataclass(frozen=True, slots=True, order=True)
class Cell:
    """One unit of harness work: a model measured on a suite at K repetitions."""

    suite: str
    model_id: str
    k: int

    def __str__(self) -> str:
        """Render as ``suite/model@K``."""
        return f"{self.suite}/{self.model_id}@K{self.k}"


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One recorded measurement of a cell.

    Args:
        cell: What was measured.
        run_id: The run that produced it.
        recorded_at: When the run was created.
        agent: Which agent drove the model.
        system: The system label the run carried.
        n_tasks: Distinct tasks with at least one recorded sample.
        n_samples: Recorded ``(task, rep)`` units.
        n_completed: Units whose agent finished without erroring or hitting the step limit.
        n_scores: Stored per-sample metric values.
        git_sha: The harness revision that produced it.
    """

    cell: Cell
    run_id: str
    recorded_at: datetime
    agent: str
    system: str
    n_tasks: int
    n_samples: int
    n_completed: int
    n_scores: int
    git_sha: str

    @property
    def expected_samples(self) -> int:
        """How many units a complete recording of this cell would hold."""
        return self.n_tasks * self.cell.k

    @property
    def is_complete(self) -> bool:
        """True when every task was recorded at the full K.

        Partial cells are common and legitimate — a free tier ran out, a laptop slept — but they
        must never be silently ranked against complete ones.
        """
        return self.n_samples >= self.expected_samples > 0

    @property
    def completion_rate(self) -> float:
        """Fraction of recorded units whose agent actually finished.

        Distinct from :attr:`is_complete`: this is about the *model* failing (crashes, step-limit
        exhaustion), not about the harness having stopped early.
        """
        if self.n_samples == 0:
            return 0.0
        return self.n_completed / self.n_samples

    @property
    def is_scored(self) -> bool:
        """True when metrics were computed, so this cell can enter a leaderboard."""
        return self.n_scores > 0


def model_of(manifest: RunManifest) -> str:
    """Return the agent-role model id recorded in a manifest.

    Falls back to ``"unknown"`` rather than guessing. A run whose model cannot be identified is
    excluded from the leaderboard instead of being attributed to the wrong model.
    """
    for reference in manifest.models:
        if reference.role == "agent":
            return reference.model_id
    return UNKNOWN_MODEL


class Ledger:
    """A read-only view of everything the harness has recorded."""

    def __init__(self, entries: list[LedgerEntry]) -> None:
        self.entries = sorted(entries, key=lambda entry: entry.recorded_at, reverse=True)

    @classmethod
    def from_store(cls, store: RunStore, *, limit: int = 10_000) -> Ledger:
        """Reconstruct the ledger from stored runs.

        Args:
            store: The run database.
            limit: Maximum runs to read, newest first.

        Returns:
            The ledger. Runs with no identifiable agent model are kept — they show up as
            ``unknown`` in the coverage view, which is the honest way to surface them.
        """
        entries: list[LedgerEntry] = []
        for row in store.list_runs(limit=limit):
            manifest = store.get_run(str(row["run_id"]))
            if manifest is None:
                continue
            entries.append(_entry_for(store, manifest))
        return cls(entries)

    # -- views ---------------------------------------------------------------

    def cells(self) -> list[Cell]:
        """Every distinct cell with at least one recording, in sorted order."""
        return sorted({entry.cell for entry in self.entries})

    def models(self) -> list[str]:
        """Every model the harness has measured."""
        return sorted({entry.cell.model_id for entry in self.entries})

    def suites(self) -> list[str]:
        """Every suite the harness has recorded against."""
        return sorted({entry.cell.suite for entry in self.entries})

    def latest(self, cell: Cell) -> LedgerEntry | None:
        """The most recent recording of ``cell``, or ``None``."""
        for entry in self.entries:
            if entry.cell == cell:
                return entry
        return None

    def history(self, cell: Cell) -> list[LedgerEntry]:
        """Every recording of ``cell``, newest first — the raw material for a trend."""
        return [entry for entry in self.entries if entry.cell == cell]

    def for_suite(self, suite: str) -> list[LedgerEntry]:
        """Latest entry per model for one suite, newest first."""
        seen: dict[str, LedgerEntry] = {}
        for entry in self.entries:
            if entry.cell.suite == suite:
                seen.setdefault(entry.cell.model_id, entry)
        return sorted(seen.values(), key=lambda entry: entry.recorded_at, reverse=True)

    def coverage(self) -> dict[str, dict[str, LedgerEntry]]:
        """``{suite: {model_id: latest entry}}`` — the coverage matrix."""
        matrix: dict[str, dict[str, LedgerEntry]] = defaultdict(dict)
        for entry in self.entries:
            matrix[entry.cell.suite].setdefault(entry.cell.model_id, entry)
        return dict(matrix)

    def missing(self, *, suites: list[str], models: list[str], k: int) -> list[Cell]:
        """Cells that have never been recorded, or were recorded only partially.

        This is the harness's work queue. A partially recorded cell is returned alongside a
        never-recorded one because both leave the same hole in the evidence, and the runner's
        resume logic makes finishing a partial cell strictly cheaper.
        """
        pending: list[Cell] = []
        for suite in suites:
            for model_id in models:
                cell = Cell(suite=suite, model_id=model_id, k=k)
                entry = self.latest(cell)
                if entry is None or not entry.is_complete or not entry.is_scored:
                    pending.append(cell)
        return pending

    def __len__(self) -> int:
        """Number of recorded entries."""
        return len(self.entries)


def _entry_for(store: RunStore, manifest: RunManifest) -> LedgerEntry:
    """Build one ledger entry by measuring what the store actually holds for a run."""
    trajectories = store.load_trajectories(manifest.run_id)
    tasks = {trajectory.task_id for trajectory in trajectories}
    completed = sum(1 for trajectory in trajectories if trajectory.status is RunStatus.COMPLETED)
    return LedgerEntry(
        cell=Cell(suite=manifest.suite.name, model_id=model_of(manifest), k=manifest.k),
        run_id=manifest.run_id,
        recorded_at=manifest.created_at,
        agent=manifest.agent,
        system=manifest.system,
        n_tasks=len(tasks),
        n_samples=len(trajectories),
        n_completed=completed,
        n_scores=store.score_count(manifest.run_id),
        git_sha=manifest.git_sha,
    )
