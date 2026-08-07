"""Persistent SQLite response cache (A3.6).

The cache is what makes CI free and reproducible: keyed by (model, messages, params, seed), it
turns a re-run into a pure function of its inputs. ``replay`` mode reads it and refuses to fall
back to the network, so a cache miss is a loud error rather than a silent quota charge.

Recorded ``latency_ms`` is replayed verbatim. Measuring SQLite lookup speed instead would make
``efficiency.latency_ms`` meaningless in exactly the mode CI uses.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from agentgate.providers.catalog import provider_of
from agentgate.providers.types import ChatRequest, ChatResponse

CACHE_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS responses (
    key               TEXT PRIMARY KEY,
    namespace         TEXT NOT NULL DEFAULT '',
    model             TEXT NOT NULL,
    provider          TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    request_json      TEXT NOT NULL,
    response_json     TEXT NOT NULL,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms        REAL NOT NULL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_responses_model ON responses(model);
CREATE INDEX IF NOT EXISTS idx_responses_namespace ON responses(namespace);
"""


class ResponseCache:
    """A write-through SQLite cache for chat completions.

    Args:
        path: File path, or ``":memory:"`` for an ephemeral cache.
        namespace: Salt mixed into every key. Bump it to invalidate a prompt revision without
            deleting the file.
    """

    def __init__(self, path: str | Path = ":memory:", *, namespace: str = "") -> None:
        self.path = str(path)
        self.namespace = namespace
        self._lock = threading.Lock()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(CACHE_SCHEMA_VERSION),),
            )

    # -- lookup ------------------------------------------------------------

    def key_for(self, request: ChatRequest) -> str:
        """Return the cache key this cache would use for ``request``."""
        return request.cache_key(self.namespace)

    def get(self, request: ChatRequest) -> ChatResponse | None:
        """Return the cached response for ``request``, or ``None`` on a miss."""
        key = self.key_for(request)
        with self._lock:
            row = self._conn.execute(
                "SELECT response_json FROM responses WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        response = ChatResponse.model_validate_json(row["response_json"])
        return response.with_cache_flag(cached=True)

    def put(self, request: ChatRequest, response: ChatResponse) -> None:
        """Store ``response`` under ``request``'s key, replacing any previous entry."""
        key = self.key_for(request)
        stored = response.with_cache_flag(cached=False)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO responses (key, namespace, model, provider, created_at,
                                       request_json, response_json,
                                       prompt_tokens, completion_tokens, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    response_json = excluded.response_json,
                    created_at = excluded.created_at,
                    prompt_tokens = excluded.prompt_tokens,
                    completion_tokens = excluded.completion_tokens,
                    latency_ms = excluded.latency_ms
                """,
                (
                    key,
                    self.namespace,
                    request.model,
                    provider_of(request.model),
                    datetime.now(UTC).isoformat(),
                    json.dumps(request.cache_payload(self.namespace), sort_keys=True),
                    stored.model_dump_json(),
                    stored.usage.prompt_tokens,
                    stored.usage.completion_tokens,
                    stored.latency_ms,
                ),
            )

    def __contains__(self, request: ChatRequest) -> bool:
        """Return True when ``request`` is cached."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM responses WHERE key = ?", (self.key_for(request),)
            ).fetchone()
        return row is not None

    # -- bulk operations ---------------------------------------------------

    def __len__(self) -> int:
        """Number of cached responses."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM responses").fetchone()
        return int(row["n"])

    def models(self) -> dict[str, int]:
        """Return ``{model: entry count}`` — used by the drift and bundle reports."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT model, COUNT(*) AS n FROM responses GROUP BY model ORDER BY model"
            ).fetchall()
        return {row["model"]: int(row["n"]) for row in rows}

    def iter_rows(self) -> Iterator[dict[str, Any]]:
        """Iterate raw cache rows in key order (for JSONL export)."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM responses ORDER BY key").fetchall()
        for row in rows:
            yield dict(row)

    def export_jsonl(self, path: str | Path) -> int:
        """Write the cache to newline-delimited JSON for shipping as a replay fixture.

        Args:
            path: Destination file.

        Returns:
            Number of rows written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with target.open("w", encoding="utf-8") as handle:
            for row in self.iter_rows():
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                count += 1
        return count

    def import_jsonl(self, path: str | Path) -> int:
        """Load rows previously written by :meth:`export_jsonl`.

        Args:
            path: Source file.

        Returns:
            Number of rows imported.
        """
        source = Path(path)
        rows: list[tuple[Any, ...]] = []
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append(
                    (
                        row["key"],
                        row.get("namespace", ""),
                        row["model"],
                        row.get("provider", provider_of(row["model"])),
                        row["created_at"],
                        row["request_json"],
                        row["response_json"],
                        row.get("prompt_tokens", 0),
                        row.get("completion_tokens", 0),
                        row.get("latency_ms", 0.0),
                    )
                )
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
        return len(rows)

    def clear(self) -> None:
        """Delete every cached response."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM responses")

    def close(self) -> None:
        """Close the underlying connection."""
        with closing(self._conn):
            pass

    def __enter__(self) -> Self:
        """Enter a context manager that closes the cache on exit."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the cache."""
        self.close()
