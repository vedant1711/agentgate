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


def merge_snapshots(existing: dict[str, Any], fresh: dict[str, Any]) -> dict[str, Any]:
    """Combine a previously committed snapshot with a newly exported one.

    **Merging rather than replacing is a correctness requirement, not a convenience.** The
    evidence base is recorded from more than one machine: local models run on a laptop that CI
    cannot reach, cloud models run on a runner that has the keys. Neither store contains the
    other's cells. An exporter that replaced the file would let whichever machine ran last delete
    everything the other had learned — and it would do so silently, since an empty snapshot is a
    perfectly valid JSON document.

    Cells are keyed by ``(suite, model, k)``. Where both sides hold the same cell the newer
    recording wins, so re-measuring a model updates it rather than duplicating it.

    Args:
        existing: The snapshot already on disk.
        fresh: The snapshot just built from this machine's store.

    Returns:
        The union, with cells sorted so the file diffs cleanly.
    """
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for cell in (*existing.get("cells", []), *fresh.get("cells", [])):
        key = (str(cell["suite"]), str(cell["model_id"]), int(cell["k"]))
        previous = by_key.get(key)
        if previous is None or str(cell["recorded_at"]) >= str(previous["recorded_at"]):
            by_key[key] = cell

    cells = [by_key[key] for key in sorted(by_key)]
    merged = dict(fresh)
    merged["cells"] = cells
    merged["n_cells"] = len(cells)
    merged["suites"] = sorted({str(cell["suite"]) for cell in cells})
    merged["models"] = sorted({str(cell["model_id"]) for cell in cells})
    return merged


def write_snapshot(
    store: RunStore, target: Path = DEFAULT_EXPORT, *, level: float = 0.95, merge: bool = True
) -> Path:
    """Write the snapshot as sorted, indented JSON so its git diffs stay readable.

    Args:
        store: The run database to export.
        target: Where to write.
        level: Confidence level for every interval.
        merge: Fold this machine's cells into whatever ``target`` already holds. On by default —
            see :func:`merge_snapshots` for why replacing is unsafe. Pass ``False`` only to
            deliberately rebuild the file from one store.
    """
    snapshot = build_snapshot(store, level=level)
    if merge and target.exists():
        try:
            previous: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
        if previous.get("cells"):
            snapshot = merge_snapshots(previous, snapshot)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target
