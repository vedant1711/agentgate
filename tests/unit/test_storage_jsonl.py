"""JSONL trajectory persistence and seed derivation."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentgate.seeds import derive_seed
from agentgate.storage import TrajectoryWriter, load_trajectories, read_trajectories
from tests.conftest import make_trajectory


def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    path = tmp_path / "trajectories.jsonl"
    originals = [make_trajectory("t1", rep=0), make_trajectory("t2", rep=1, system="candidate")]
    with TrajectoryWriter(path) as writer:
        assert writer.write_all(originals) == 2
    restored = load_trajectories(path)
    assert restored == originals


def test_writer_flushes_so_an_interrupted_run_keeps_its_work(tmp_path: Path) -> None:
    path = tmp_path / "partial.jsonl"
    writer = TrajectoryWriter(path)
    writer.write(make_trajectory("t1"))
    assert len(load_trajectories(path)) == 1, "written before close"
    writer.close()


def test_append_mode_continues_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "resume.jsonl"
    with TrajectoryWriter(path) as writer:
        writer.write(make_trajectory("t1"))
    with TrajectoryWriter(path, append=True) as writer:
        writer.write(make_trajectory("t2"))
    assert [t.task_id for t in read_trajectories(path)] == ["t1", "t2"]


def test_truncated_final_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "torn.jsonl"
    with TrajectoryWriter(path) as writer:
        writer.write(make_trajectory("t1"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"task_id": "t2", "re')  # killed mid-write
    assert [t.task_id for t in read_trajectories(path)] == ["t1"]


def test_missing_file_yields_nothing(tmp_path: Path) -> None:
    assert load_trajectories(tmp_path / "absent.jsonl") == []


def test_writer_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "out.jsonl"
    with TrajectoryWriter(path) as writer:
        writer.write(make_trajectory())
    assert path.exists()


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def test_derived_seeds_are_deterministic() -> None:
    assert derive_seed(42, "task-1", 0) == derive_seed(42, "task-1", 0)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ((42, "task-1", 0), (42, "task-1", 1)),
        ((42, "task-1", 0), (42, "task-2", 0)),
        ((42, "task-1", 0), (43, "task-1", 0)),
    ],
)
def test_different_identities_produce_different_seeds(
    a: tuple[int, str, int], b: tuple[int, str, int]
) -> None:
    assert derive_seed(a[0], a[1], a[2]) != derive_seed(b[0], b[1], b[2])


def test_seeds_are_non_negative_and_bounded() -> None:
    for rep in range(50):
        seed = derive_seed(7, "task", rep)
        assert 0 <= seed < 2**60
