"""DuckDB persistence for runs and their per-sample records.

Single-file, zero-ops, and fast enough for millions of rows — which matters because the
statistics engine and the demo Playground both need **per-sample** scores, not aggregates. An
aggregate cannot be re-analysed at a different K, a different margin, or a different alpha, and
re-analysis is the entire point of the interactive demo (K2).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import duckdb

from agentgate.schemas.results import MetricResult, RunManifest
from agentgate.schemas.trajectory import Trajectory

STORE_SCHEMA_VERSION = 1
DEFAULT_STORE_PATH = Path(".agentgate/agentgate.duckdb")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id        VARCHAR PRIMARY KEY,
    created_at    TIMESTAMP NOT NULL,
    suite_name    VARCHAR NOT NULL,
    suite_version VARCHAR NOT NULL,
    suite_hash    VARCHAR NOT NULL,
    system        VARCHAR NOT NULL,
    agent         VARCHAR NOT NULL,
    k             INTEGER NOT NULL,
    base_seed     BIGINT  NOT NULL,
    mode          VARCHAR NOT NULL,
    config_hash   VARCHAR NOT NULL,
    git_sha       VARCHAR NOT NULL,
    faults        VARCHAR NOT NULL,
    manifest_json VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS samples (
    run_id            VARCHAR NOT NULL,
    task_id           VARCHAR NOT NULL,
    rep               INTEGER NOT NULL,
    cluster_id        VARCHAR NOT NULL,
    seed              BIGINT  NOT NULL,
    status            VARCHAR NOT NULL,
    latency_ms        DOUBLE  NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    tool_calls        INTEGER NOT NULL,
    est_cost_usd      DOUBLE  NOT NULL,
    trajectory_json   VARCHAR NOT NULL,
    PRIMARY KEY (run_id, task_id, rep)
);

CREATE TABLE IF NOT EXISTS scores (
    run_id      VARCHAR NOT NULL,
    task_id     VARCHAR NOT NULL,
    rep         INTEGER NOT NULL,
    metric      VARCHAR NOT NULL,
    value       DOUBLE,
    family      VARCHAR NOT NULL,
    dtype       VARCHAR NOT NULL,
    direction   VARCHAR NOT NULL,
    status      VARCHAR NOT NULL,
    cost_tokens INTEGER NOT NULL DEFAULT 0,
    result_json VARCHAR NOT NULL,
    PRIMARY KEY (run_id, task_id, rep, metric)
);
CREATE INDEX IF NOT EXISTS idx_scores_metric ON scores(run_id, metric);
"""


def _initialise(target: Path) -> None:
    """Create an empty database with the schema in place, then release it."""
    connection = duckdb.connect(str(target))
    try:
        connection.execute(_SCHEMA)
    finally:
        connection.close()


class RunStore:
    """Read/write access to the run database.

    Args:
        path: DuckDB file, or ``":memory:"`` for an ephemeral store.
        read_only: Open without acquiring a write lock (for the demo and report renderers).
    """

    def __init__(self, path: str | Path = DEFAULT_STORE_PATH, *, read_only: bool = False) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            target = Path(self.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if read_only and not target.exists():
                # Initialise properly rather than touching an empty file: DuckDB rejects a
                # zero-byte database, so a bare touch produces exactly the failure it was meant
                # to avoid. Opening read-write once creates a real, empty, schema-bearing file.
                _initialise(target)
        self._conn = duckdb.connect(self.path, read_only=read_only)
        if not read_only:
            self._conn.execute(_SCHEMA)
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT (key) DO NOTHING",
                [str(STORE_SCHEMA_VERSION)],
            )

    # -- runs --------------------------------------------------------------

    def save_run(self, manifest: RunManifest) -> None:
        """Insert or replace a run's manifest row."""
        self._conn.execute("DELETE FROM runs WHERE run_id = ?", [manifest.run_id])
        self._conn.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                manifest.run_id,
                manifest.created_at,
                manifest.suite.name,
                manifest.suite.version,
                manifest.suite.content_hash,
                manifest.system,
                manifest.agent,
                manifest.k,
                manifest.base_seed,
                manifest.mode.value,
                manifest.config_hash,
                manifest.git_sha,
                ",".join(manifest.faults),
                manifest.model_dump_json(),
            ],
        )

    def get_run(self, run_id: str) -> RunManifest | None:
        """Return a stored manifest, or ``None`` when the run is unknown."""
        row = self._conn.execute(
            "SELECT manifest_json FROM runs WHERE run_id = ?", [run_id]
        ).fetchone()
        return None if row is None else RunManifest.model_validate_json(row[0])

    def list_runs(self, *, suite: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """List runs newest first, optionally filtered by suite name."""
        sql = (
            "SELECT run_id, created_at, suite_name, system, agent, k, mode, faults "
            "FROM runs {where} ORDER BY created_at DESC LIMIT ?"
        )
        params: list[Any] = []
        where = ""
        if suite is not None:
            where = "WHERE suite_name = ?"
            params.append(suite)
        params.append(limit)
        cursor = self._conn.execute(sql.format(where=where), params)
        columns = [description[0] for description in cursor.description or []]
        return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

    def latest_run(self, suite: str, system: str) -> str | None:
        """Return the most recent run id for ``(suite, system)``."""
        row = self._conn.execute(
            "SELECT run_id FROM runs WHERE suite_name = ? AND system = ? "
            "ORDER BY created_at DESC LIMIT 1",
            [suite, system],
        ).fetchone()
        return None if row is None else str(row[0])

    def delete_run(self, run_id: str) -> None:
        """Remove a run with its samples and scores."""
        self._conn.execute("DELETE FROM scores WHERE run_id = ?", [run_id])
        self._conn.execute("DELETE FROM samples WHERE run_id = ?", [run_id])
        self._conn.execute("DELETE FROM runs WHERE run_id = ?", [run_id])

    # -- samples -----------------------------------------------------------

    def save_samples(
        self, run_id: str, trajectories: Iterable[Trajectory], clusters: dict[str, str]
    ) -> int:
        """Persist trajectories for a run.

        Args:
            run_id: Owning run.
            trajectories: Records to store; existing (task, rep) rows are replaced.
            clusters: ``{task_id: cluster_id}``, denormalised so clustered SEs can be computed
                without re-reading the suite.

        Returns:
            Number of rows written.
        """
        rows = [
            [
                run_id,
                trajectory.task_id,
                trajectory.rep,
                clusters.get(trajectory.task_id, trajectory.task_id),
                trajectory.seed,
                trajectory.status.value,
                trajectory.latency_ms,
                trajectory.usage.prompt_tokens,
                trajectory.usage.completion_tokens,
                trajectory.n_tool_calls,
                trajectory.est_cost_usd,
                trajectory.model_dump_json(),
            ]
            for trajectory in trajectories
        ]
        for row in rows:
            self._conn.execute(
                "DELETE FROM samples WHERE run_id = ? AND task_id = ? AND rep = ?",
                [row[0], row[1], row[2]],
            )
            self._conn.execute(
                "INSERT INTO samples VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row
            )
        return len(rows)

    # -- scores ------------------------------------------------------------

    def save_scores(self, run_id: str, results: Iterable[MetricResult]) -> int:
        """Persist per-sample metric results.

        Per-sample rows, never aggregates: an aggregate cannot be re-analysed at a different K,
        margin, or alpha, and that re-analysis is what the gate and the demo both do.

        Args:
            run_id: Owning run.
            results: Metric results to store; existing rows for the same key are replaced.

        Returns:
            Number of rows written.
        """
        rows = [
            [
                run_id,
                result.task_id,
                result.rep,
                result.metric,
                result.value,
                result.family.value,
                result.dtype,
                result.direction,
                result.status,
                result.cost_tokens,
                result.model_dump_json(),
            ]
            for result in results
        ]
        for row in rows:
            self._conn.execute(
                "DELETE FROM scores WHERE run_id = ? AND task_id = ? AND rep = ? AND metric = ?",
                [row[0], row[1], row[2], row[3]],
            )
            self._conn.execute("INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)
        return len(rows)

    def load_scores(self, run_id: str, *, metric: str | None = None) -> list[MetricResult]:
        """Return stored metric results for a run, optionally filtered to one metric."""
        sql = "SELECT result_json FROM scores WHERE run_id = ?"
        params: list[Any] = [run_id]
        if metric is not None:
            sql += " AND metric = ?"
            params.append(metric)
        sql += " ORDER BY metric, task_id, rep"
        rows = self._conn.execute(sql, params).fetchall()
        return [MetricResult.model_validate_json(row[0]) for row in rows]

    def scored_metric_names(self, run_id: str) -> list[str]:
        """Metric names with at least one usable value in this run."""
        rows = self._conn.execute(
            "SELECT DISTINCT metric FROM scores WHERE run_id = ? AND status = 'ok' "
            "AND value IS NOT NULL ORDER BY metric",
            [run_id],
        ).fetchall()
        return [str(row[0]) for row in rows]

    def score_count(self, run_id: str) -> int:
        """Number of stored metric results for a run."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM scores WHERE run_id = ?", [run_id]
        ).fetchone()
        return 0 if row is None else int(row[0])

    def load_trajectories(self, run_id: str) -> list[Trajectory]:
        """Return every trajectory for ``run_id`` in (task, rep) order."""
        rows = self._conn.execute(
            "SELECT trajectory_json FROM samples WHERE run_id = ? ORDER BY task_id, rep",
            [run_id],
        ).fetchall()
        return [Trajectory.model_validate_json(row[0]) for row in rows]

    def iter_trajectories(self, run_id: str) -> Iterator[Trajectory]:
        """Stream trajectories for ``run_id``."""
        yield from self.load_trajectories(run_id)

    def completed_units(self, run_id: str) -> set[tuple[str, int]]:
        """Return the ``(task_id, rep)`` pairs already recorded — the resume frontier."""
        rows = self._conn.execute(
            "SELECT task_id, rep FROM samples WHERE run_id = ?", [run_id]
        ).fetchall()
        return {(str(task_id), int(rep)) for task_id, rep in rows}

    def clusters_for(self, run_id: str) -> dict[str, str]:
        """Return ``{task_id: cluster_id}`` as recorded for ``run_id``."""
        rows = self._conn.execute(
            "SELECT DISTINCT task_id, cluster_id FROM samples WHERE run_id = ?", [run_id]
        ).fetchall()
        return {str(task_id): str(cluster_id) for task_id, cluster_id in rows}

    def sample_count(self, run_id: str) -> int:
        """Number of stored samples for a run."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM samples WHERE run_id = ?", [run_id]
        ).fetchone()
        return 0 if row is None else int(row[0])

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the connection."""
        self._conn.close()

    def __enter__(self) -> Self:
        """Enter a context manager that closes the store on exit."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the store."""
        self.close()
