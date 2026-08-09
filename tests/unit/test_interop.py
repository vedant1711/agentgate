"""Export tests: the parity claim has to be checkable, not just asserted.

This module exists so someone with an API key can score the same trajectories with RAGAS or
DeepEval and compare per sample. That only works if the export is faithful — every recorded
repetition present, no invented ground truth, and each library's own field names.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentgate.errors import ConfigError
from agentgate.interop import FORMATS, export_run, rows_for, write_rows
from agentgate.schemas.task import ReferenceSpec, SuiteSpec, TaskSpec
from agentgate.schemas.trajectory import FinalStep, Trajectory
from agentgate.storage.duckdb_store import RunStore
from tests.conftest import make_manifest, make_suite, make_trajectory


def suite_with_contexts() -> SuiteSpec:
    """A one-task suite whose task has a question and a reference answer."""
    base = make_suite()
    task = TaskSpec(
        id="q1",
        cluster_id="c1",
        inputs={"question": "What is the refund window?"},
        reference=ReferenceSpec(answer="30 days"),
    )
    return base.model_copy(update={"tasks": [task]})


def answered(task_id: str = "q1", *, rep: int = 0, contexts: tuple[str, ...] = ()) -> Trajectory:
    trajectory = Trajectory(task_id=task_id, rep=rep, seed=1, system="baseline")
    trajectory.add_step(FinalStep(index=0, answer="Thirty days."))
    trajectory.final_answer = "Thirty days."
    if contexts:
        trajectory.retrieved_contexts = list(contexts)
    return trajectory


def test_ragas_rows_use_ragas_field_names() -> None:
    suite = suite_with_contexts()
    rows = list(rows_for(suite, [answered(contexts=("policy: 30 days",))], fmt="ragas"))
    assert len(rows) == 1
    row = rows[0]
    assert row["question"] == "What is the refund window?"
    assert row["answer"] == "Thirty days."
    assert row["contexts"] == ["policy: 30 days"]
    # RAGAS renamed ground_truth to reference; emitting both keeps old and new versions working.
    assert row["ground_truth"] == row["reference"] == "30 days"


def test_deepeval_rows_use_deepeval_field_names() -> None:
    rows = list(rows_for(suite_with_contexts(), [answered()], fmt="deepeval"))
    assert set(rows[0]) >= {"input", "actual_output", "expected_output", "retrieval_context"}
    assert rows[0]["input"] == "What is the refund window?"
    assert rows[0]["expected_output"] == "30 days"


def test_the_neutral_format_keeps_what_the_others_drop() -> None:
    """Cluster and seed are what make a paired, clustered re-analysis possible downstream."""
    rows = list(rows_for(suite_with_contexts(), [answered()], fmt="jsonl"))
    assert set(rows[0]) >= {"cluster_id", "seed", "status", "tools"}


def test_a_task_without_a_reference_exports_an_empty_ground_truth() -> None:
    """Empty, so reference-requiring metrics skip the row rather than grading against a fiction."""
    suite = make_suite().model_copy(
        update={"tasks": [TaskSpec(id="q1", cluster_id="c1", inputs={"question": "Why?"})]}
    )
    rows = list(rows_for(suite, [answered()], fmt="ragas"))
    assert rows[0]["ground_truth"] == ""


def test_every_repetition_is_exported_not_just_one_per_task() -> None:
    """Collapsing K to one row would destroy the per-sample comparison this file exists for."""
    suite = suite_with_contexts()
    rows = list(rows_for(suite, [answered(rep=rep) for rep in range(4)], fmt="ragas"))
    assert [row["rep"] for row in rows] == [0, 1, 2, 3]


def test_trajectories_for_unknown_tasks_are_skipped_not_guessed() -> None:
    rows = list(rows_for(suite_with_contexts(), [answered("not-in-suite")], fmt="ragas"))
    assert rows == []


def test_an_unknown_format_is_rejected_by_name() -> None:
    with pytest.raises(ConfigError, match="unknown export format"):
        list(rows_for(suite_with_contexts(), [answered()], fmt="parquet"))  # type: ignore[arg-type]


def test_every_declared_format_produces_rows() -> None:
    suite = suite_with_contexts()
    for fmt in FORMATS:
        assert list(rows_for(suite, [answered()], fmt=fmt))  # type: ignore[arg-type]


def test_written_rows_are_one_json_object_per_line(tmp_path: Path) -> None:
    target = tmp_path / "out.jsonl"
    suite = suite_with_contexts()
    written = write_rows(rows_for(suite, [answered(rep=r) for r in range(3)], fmt="ragas"), target)
    assert written == 3
    lines = target.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert all(json.loads(line)["question"] for line in lines)


def test_exporting_an_unknown_run_fails_loudly(tmp_path: Path) -> None:
    """An empty file would look like a successful export of nothing."""
    with (
        RunStore(tmp_path / "s.duckdb") as store,
        pytest.raises(ConfigError, match="nothing to export"),
    ):
        export_run(
            store,
            run_id="ghost",
            suite=suite_with_contexts(),
            target=tmp_path / "out.jsonl",
        )


def test_exporting_a_stored_run_round_trips(tmp_path: Path) -> None:
    suite = make_suite()
    with RunStore(tmp_path / "s.duckdb") as store:
        store.save_run(make_manifest("r1", suite=suite))
        trajectories = [make_trajectory(task.id) for task in suite.tasks]
        store.save_samples("r1", trajectories, {t.id: t.cluster_id for t in suite.tasks})

        target = tmp_path / "out.jsonl"
        written = export_run(store, run_id="r1", suite=suite, target=target, fmt="jsonl")

    assert written == len(suite.tasks)
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert {row["task_id"] for row in rows} == {task.id for task in suite.tasks}
