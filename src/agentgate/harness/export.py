"""A compact, committable snapshot of what the harness knows.

The run store is the source of truth, and it is not committed: it is tens of megabytes of full
trajectories, it grows without bound, and it is regenerable from the provider cache. What *is*
worth committing is the derived summary — small, diffable, and the thing a reader or a docs site
actually wants.

Because it is committed, every recording session produces a reviewable diff of what the project
learned. That is the mechanism by which "this project improves itself over time" is an auditable
claim rather than a slogan: the evidence base's history is the repository's history.

Two rules keep the snapshot honest:

* **Every estimate carries its interval and its n.** A bare number in a JSON file is exactly the
  artifact this project exists to argue against, and it is the one most likely to be copied into
  a slide without its uncertainty.
* **Skipped metrics are omitted, never zero-filled.** A metric that did not apply to a suite is
  absent, because writing 0.0 would make a category error look like a catastrophic score.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentgate import __version__
from agentgate.harness.ledger import Ledger
from agentgate.providers.models import get_card
from agentgate.stats.aggregate import summarise_metric

if TYPE_CHECKING:
    from agentgate.storage.duckdb_store import RunStore

DEFAULT_EXPORT = Path("results/harness.json")
SNAPSHOT_VERSION = 1


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    """One metric's value for one recording, with the uncertainty it cannot be read without."""

    metric: str
    value: float
    ci_low: float | None
    ci_high: float | None
    n: int
    method: str
    direction: str
    dtype: str


@dataclass(frozen=True, slots=True)
class CellSnapshot:
    """Everything known about one model on one suite."""

    suite: str
    model_id: str
    label: str
    k: int
    run_id: str
    recorded_at: str
    agent: str
    n_tasks: int
    n_samples: int
    completion_rate: float
    complete: bool
    git_sha: str
    metrics: list[MetricSnapshot]


def build_snapshot(
    store: RunStore, *, ledger: Ledger | None = None, level: float = 0.95
) -> dict[str, Any]:
    """Summarise every recording in the store.

    Args:
        store: The run database.
        ledger: Prebuilt ledger.
        level: Confidence level for every interval.

    Returns:
        A JSON-ready dictionary, with cells sorted so the file diffs cleanly between sessions.
    """
    ledger = ledger or Ledger.from_store(store)
    cells: list[CellSnapshot] = []

    for entry in sorted(ledger.entries, key=lambda item: (item.cell.suite, item.cell.model_id)):
        clusters = store.clusters_for(entry.run_id)
        metrics: list[MetricSnapshot] = []
        for name in store.scored_metric_names(entry.run_id):
            summary = summarise_metric(
                store.load_scores(entry.run_id, metric=name), clusters=clusters, level=level
            )
            if summary is None:
                continue
            estimate = summary.clustered or summary.estimate
            metrics.append(
                MetricSnapshot(
                    metric=name,
                    value=estimate.value,
                    ci_low=estimate.ci_low,
                    ci_high=estimate.ci_high,
                    n=estimate.n,
                    method=estimate.method,
                    direction=summary.direction,
                    dtype=summary.dtype,
                )
            )

        card = get_card(entry.cell.model_id)
        cells.append(
            CellSnapshot(
                suite=entry.cell.suite,
                model_id=entry.cell.model_id,
                label=card.label if card else entry.cell.model_id,
                k=entry.cell.k,
                run_id=entry.run_id,
                recorded_at=entry.recorded_at.isoformat(),
                agent=entry.agent,
                n_tasks=entry.n_tasks,
                n_samples=entry.n_samples,
                completion_rate=round(entry.completion_rate, 4),
                complete=entry.is_complete,
                git_sha=entry.git_sha,
                metrics=metrics,
            )
        )

    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "agentgate_version": __version__,
        "ci_level": level,
        "note": (
            "Every value carries its confidence interval and n. Metrics that did not apply to a "
            "suite are omitted rather than scored zero. Cells marked complete:false were only "
            "partially recorded and are not comparable to complete ones."
        ),
        "n_cells": len(cells),
        "suites": sorted({cell.suite for cell in cells}),
        "models": sorted({cell.model_id for cell in cells}),
        "cells": [asdict(cell) for cell in cells],
    }


def write_snapshot(store: RunStore, target: Path = DEFAULT_EXPORT, *, level: float = 0.95) -> Path:
    """Write the snapshot as sorted, indented JSON so its git diffs stay readable."""
    snapshot = build_snapshot(store, level=level)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target
