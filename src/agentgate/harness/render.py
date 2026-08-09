"""Render the evidence snapshot as a documentation page.

Generated rather than hand-written, for the same reason the metric catalogue is: a results page
maintained by hand drifts from the results, and a stale number presented confidently is worse than
no number. ``agentgate docs results --check`` fails CI when the committed page no longer matches
the committed snapshot.

Every table here prints the interval next to the value. That is not decoration — a page that
listed bare point estimates would be the exact artifact this project exists to argue against, and
it would be the one most likely to be screenshotted without its uncertainty.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agentgate.harness.export import DEFAULT_EXPORT

HEADLINE_METRICS = (
    "outcome.task_success",
    "judge.instruction_following",
    "judge.coherence",
    "trajectory.recall",
    "trajectory.precision",
    "trajectory.argument_correctness",
    "trajectory.error_recovery_rate",
    "trajectory.step_efficiency",
)
"""Metrics worth leading with. The rest are still rendered, just further down."""

_PREAMBLE = """# Current results

Everything on this page is generated from `results/harness.json`, which is written by
`agentgate harness export` and committed on every recording session. Nothing here is typed by
hand, so it cannot drift from the evidence it describes.

**Read every value with its interval.** A point estimate on its own is the artifact this project
exists to argue against.
"""

_CAVEAT = """
!!! note "What these numbers are not"

    The τ² suite is a **single-turn adaptation** of τ²-bench, which is multi-turn with a simulated
    user. The tasks, the tool surface and the gold trajectories are τ²-bench's; the interaction
    protocol is ours. These are **not** τ²-bench leaderboard scores and must not be compared to
    them.

    Metrics absent from a table did not apply to that suite and were skipped, never scored zero.
"""


def load_snapshot(path: Path = DEFAULT_EXPORT) -> dict[str, Any]:
    """Read a committed snapshot.

    Raises:
        FileNotFoundError: When no snapshot has been exported yet, naming the fix.
    """
    if not path.exists():
        msg = f"no snapshot at {path}. Run `agentgate harness export` first."
        raise FileNotFoundError(msg)
    parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return parsed


def _interval(metric: dict[str, Any]) -> str:
    """Render a metric's interval, or say plainly that it has none."""
    low, high = metric.get("ci_low"), metric.get("ci_high")
    if low is None or high is None:
        return "—"
    return f"[{low:.3f}, {high:.3f}]"


def _value(metric: dict[str, Any]) -> str:
    """Render a metric value at a precision suited to its scale."""
    value = float(metric["value"])
    return f"{value:.3f}" if abs(value) < 1000 else f"{value:,.0f}"


def _metric_table(metrics: list[dict[str, Any]], names: tuple[str, ...] | None = None) -> list[str]:
    """Render metrics as a markdown table, in the order given or alphabetically."""
    by_name = {metric["metric"]: metric for metric in metrics}
    chosen = (
        [by_name[name] for name in names if name in by_name]
        if names is not None
        else sorted(metrics, key=lambda metric: str(metric["metric"]))
    )
    if not chosen:
        return []
    lines = ["| metric | value | 95% CI | n | method |", "|---|---|---|---|---|"]
    lines.extend(
        f"| `{metric['metric']}` | {_value(metric)} | {_interval(metric)} "
        f"| {metric['n']} | {metric['method']} |"
        for metric in chosen
    )
    return lines


def render_results(snapshot: dict[str, Any]) -> str:
    """Turn a snapshot into the results documentation page."""
    lines = [_PREAMBLE]

    cells = snapshot.get("cells", [])
    if not cells:
        lines.append(
            "\nNo model has been recorded yet. Run `agentgate harness record` to start the "
            "evidence base.\n"
        )
        return "\n".join(lines)

    lines.append(
        f"\n**{len(cells)} recording(s)** across {len(snapshot['suites'])} suite(s) "
        f"and {len(snapshot['models'])} model(s), at "
        f"{float(snapshot.get('ci_level', 0.95)) * 100:.0f}% confidence.\n"
    )
    lines.append(_CAVEAT)

    for cell in cells:
        partial = "" if cell["complete"] else " · ⚠️ **partial recording**"
        lines.append(f"\n## {cell['label']} on `{cell['suite']}`{partial}\n")
        lines.append(
            f"`{cell['model_id']}` · agent `{cell['agent']}` "
            f"· {cell['n_tasks']} tasks x K={cell['k']} = {cell['n_samples']} units "
            f"· {float(cell['completion_rate']) * 100:.0f}% completed "
            f"· recorded {cell['recorded_at'][:10]}\n"
        )
        if not cell["complete"]:
            lines.append(
                "\n!!! warning\n\n    This cell was only partially recorded, so it is excluded "
                "from leaderboards. A model measured on a subset of tasks is not comparable to "
                "one measured on all of them.\n"
            )

        metrics = cell["metrics"]
        headline = _metric_table(metrics, HEADLINE_METRICS)
        if headline:
            lines.append("\n### Headline\n")
            lines.extend(headline)

        rest = [metric for metric in metrics if metric["metric"] not in set(HEADLINE_METRICS)]
        remaining = _metric_table(rest)
        if remaining:
            lines.append('\n??? note "All other metrics"\n')
            lines.extend(f"    {line}" for line in remaining)

    lines.append("")
    return "\n".join(lines)


def write_results(
    snapshot_path: Path = DEFAULT_EXPORT, target: Path = Path("docs/results.md")
) -> Path:
    """Generate the results page from the committed snapshot."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_results(load_snapshot(snapshot_path)), encoding="utf-8")
    return target


def results_are_current(
    snapshot_path: Path = DEFAULT_EXPORT, target: Path = Path("docs/results.md")
) -> bool:
    """True when the committed page matches what the snapshot would generate."""
    if not target.exists():
        return False
    return target.read_text(encoding="utf-8") == render_results(load_snapshot(snapshot_path))
