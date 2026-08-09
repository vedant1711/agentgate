"""Export recorded runs into the shapes other evaluation libraries expect.

AgentGate reimplements metrics that RAGAS and DeepEval also provide, and a claim that two
implementations measure the same thing is worth nothing unless someone can check it. This module
exists so they can: it writes a recorded run out in each library's input format, so the same
trajectories can be scored twice and compared **per sample**.

Per sample matters. Two implementations can agree on the suite mean while disagreeing on every
individual item, and only the item-level correlation distinguishes "same measurement" from
"same average by coincidence."

Nothing here calls a model. The export is a pure projection of what was already recorded, so it
runs offline, costs nothing, and produces the same file every time.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from agentgate.errors import ConfigError
from agentgate.schemas.task import SuiteSpec, TaskSpec

if TYPE_CHECKING:
    from agentgate.schemas.trajectory import Trajectory
    from agentgate.storage.duckdb_store import RunStore

ExportFormat = Literal["ragas", "deepeval", "jsonl"]

FORMATS: tuple[str, ...] = ("ragas", "deepeval", "jsonl")


def _question(task: TaskSpec, trajectory: Trajectory) -> str:
    """The prompt the agent was given, however the suite spelled it."""
    for key in ("question", "prompt", "instruction", "query"):
        value = task.inputs.get(key)
        if isinstance(value, str) and value.strip():
            return value
    fallback = trajectory.metadata.get("prompt", "")
    return fallback if isinstance(fallback, str) else ""


def _ground_truth(task: TaskSpec) -> str:
    """The reference answer, or an empty string when the suite declares none.

    Empty rather than omitted: RAGAS metrics that need a reference will skip the row, which is the
    correct behaviour. Inventing a reference to keep the column populated would silently change
    what the comparison measures.
    """
    return task.reference.answer or ""


def rows_for(
    suite: SuiteSpec, trajectories: list[Trajectory], *, fmt: ExportFormat = "ragas"
) -> Iterator[dict[str, Any]]:
    """Project trajectories into one dictionary per (task, repetition).

    Args:
        suite: The suite that was run, for prompts and references.
        trajectories: Recorded trajectories.
        fmt: Output shape. ``ragas`` and ``deepeval`` use each library's field names; ``jsonl``
            is the neutral superset.

    Yields:
        One row per trajectory, keyed for the requested library.

    Raises:
        ConfigError: When ``fmt`` is not a known format.
    """
    if fmt not in FORMATS:
        msg = f"unknown export format {fmt!r}; known formats: {', '.join(FORMATS)}"
        raise ConfigError(msg)

    tasks = {task.id: task for task in suite.tasks}
    for trajectory in trajectories:
        task = tasks.get(trajectory.task_id)
        if task is None:
            continue
        question = _question(task, trajectory)
        contexts = list(trajectory.retrieved_contexts)
        truth = _ground_truth(task)

        if fmt == "ragas":
            # RAGAS reads these exact keys; `reference` is its newer name for ground truth.
            yield {
                "question": question,
                "answer": trajectory.final_answer,
                "contexts": contexts,
                "ground_truth": truth,
                "reference": truth,
                "task_id": trajectory.task_id,
                "rep": trajectory.rep,
            }
        elif fmt == "deepeval":
            yield {
                "input": question,
                "actual_output": trajectory.final_answer,
                "expected_output": truth,
                "retrieval_context": contexts,
                "task_id": trajectory.task_id,
                "rep": trajectory.rep,
            }
        else:
            yield {
                "task_id": trajectory.task_id,
                "cluster_id": task.cluster_id,
                "rep": trajectory.rep,
                "seed": trajectory.seed,
                "status": trajectory.status.value,
                "question": question,
                "answer": trajectory.final_answer,
                "contexts": contexts,
                "ground_truth": truth,
                "tools": trajectory.tool_sequence,
            }


def write_rows(rows: Iterator[dict[str, Any]], target: Path) -> int:
    """Write rows as JSON Lines and return how many were written."""
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def export_run(
    store: RunStore,
    *,
    run_id: str,
    suite: SuiteSpec,
    target: Path,
    fmt: ExportFormat = "ragas",
) -> int:
    """Export one stored run to ``target``.

    Args:
        store: The run database.
        run_id: Which run to export.
        suite: The suite it was run against.
        target: Output file.
        fmt: Output shape.

    Returns:
        Rows written.

    Raises:
        ConfigError: When the run is unknown, rather than writing an empty file that looks like
            a successful export of nothing.
    """
    trajectories = store.load_trajectories(run_id)
    if not trajectories:
        msg = f"run {run_id!r} has no stored trajectories; nothing to export"
        raise ConfigError(msg)
    return write_rows(rows_for(suite, trajectories, fmt=fmt), target)
