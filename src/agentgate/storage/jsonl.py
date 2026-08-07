"""JSONL persistence for trajectories.

Trajectories are always written as newline-delimited JSON, even when OpenTelemetry export is
configured, because the metrics engine must never depend on a collector being up. One line per
(task, repetition) keeps the file streamable, appendable, and resumable — a partially written
run is still a valid prefix.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from types import TracebackType
from typing import IO, Self

from agentgate.schemas.trajectory import Trajectory


class TrajectoryWriter:
    """Append-only JSONL writer.

    Args:
        path: Destination file; parent directories are created.
        append: Continue an existing file (used by resume) instead of truncating.
    """

    def __init__(self, path: str | Path, *, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: IO[str] = self.path.open("a" if append else "w", encoding="utf-8")
        self.count = 0

    def write(self, trajectory: Trajectory) -> None:
        """Write one trajectory and flush, so an interrupted run loses nothing."""
        self._handle.write(trajectory.model_dump_json() + "\n")
        self._handle.flush()
        self.count += 1

    def write_all(self, trajectories: Iterable[Trajectory]) -> int:
        """Write several trajectories.

        Args:
            trajectories: Records to write.

        Returns:
            Number written.
        """
        written = 0
        for trajectory in trajectories:
            self.write(trajectory)
            written += 1
        return written

    def close(self) -> None:
        """Close the file handle."""
        self._handle.close()

    def __enter__(self) -> Self:
        """Enter a context manager that closes the file on exit."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the writer."""
        self.close()


def read_trajectories(path: str | Path) -> Iterator[Trajectory]:
    """Stream trajectories from a JSONL file.

    Args:
        path: Source file.

    Yields:
        Each recorded trajectory, in file order. Blank lines are skipped; a truncated final line
        (an interrupted run) is ignored rather than fatal.
    """
    source = Path(path)
    if not source.exists():
        return
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            yield Trajectory.model_validate(payload)


def load_trajectories(path: str | Path) -> list[Trajectory]:
    """Read every trajectory from ``path`` into memory."""
    return list(read_trajectories(path))
