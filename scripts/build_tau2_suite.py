"""Convert tau2-bench retail tasks into an AgentGate suite.

tau2-bench (MIT, sierra-research/tau2-bench) ships 114 retail tasks, 112 of which carry gold
tool-call sequences under ``evaluation_criteria.actions``. Those map almost directly onto
AgentGate's reference-trajectory format — tool name plus expected arguments — which is what makes
this adoption honest rather than approximate: we are using *their* ground truth, not inventing
ours.

**The one real difference, stated loudly.** tau2-bench is multi-turn: an agent converses with a
simulated user who withholds information until asked. AgentGate is single-turn. So each task's
user scenario is flattened into one instruction carrying what the user knows up front, and
``unknown_info`` is omitted — the agent must still discover it through tools, which is the part
that actually exercises the trajectory metrics.

Scores from this suite are therefore **not** comparable to a tau2-bench leaderboard, and the
generated suite says so in its own description.

Usage::

    uv run python scripts/build_tau2_suite.py --download   # fetch the data first
    uv run python scripts/build_tau2_suite.py
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

import yaml

DATA = Path("datasets/tau2")
OUTPUT = Path("suites/tau2_retail/suite.yaml")
RAW_BASE = "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains"

# Tasks whose gold sequence is a single hand-off carry no tool-use signal worth gating on.
TRIVIAL_ONLY = {"transfer_to_human_agents"}


def download() -> None:
    """Fetch the retail database and tasks from the tau2-bench repository."""
    DATA.mkdir(parents=True, exist_ok=True)
    for name in ("db.json", "tasks.json"):
        target = DATA / f"retail-{name}"
        url = f"{RAW_BASE}/retail/{name}"
        print(f"fetching {url}")
        with urllib.request.urlopen(url) as response:
            target.write_bytes(response.read())
        print(f"  wrote {target} ({target.stat().st_size:,} bytes)")


def instruction_for(task: dict[str, Any]) -> str:
    """Flatten a multi-turn user scenario into a single instruction.

    ``known_info`` is included because a real user volunteers it; ``unknown_info`` is deliberately
    withheld, so the agent still has to look things up rather than being handed the answer.
    """
    scenario = (task.get("user_scenario") or {}).get("instructions") or {}
    reason = str(scenario.get("reason_for_call") or "").strip()
    known = str(scenario.get("known_info") or "").strip()
    parts = [part for part in (known, reason) if part]
    return " ".join(parts) or "Assist the customer with their request."


def reference_steps(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Turn tau2's gold actions into AgentGate reference steps."""
    actions = (task.get("evaluation_criteria") or {}).get("actions") or []
    steps: list[dict[str, Any]] = []
    for action in actions:
        name = action.get("name")
        if not name:
            continue
        step: dict[str, Any] = {"tools": [name]}
        arguments = action.get("arguments") or {}
        # Only scalar arguments are compared. tau2 encodes list arguments (item swaps) in an
        # order the agent has no way to know, so pinning them would penalise correct behaviour.
        scalars = {
            key: value
            for key, value in arguments.items()
            if isinstance(value, str | int | float | bool)
        }
        if scalars:
            step["args"] = scalars
        steps.append(step)
    return steps


def convert(limit: int | None = None) -> dict[str, Any]:
    """Build the suite document from the vendored tau2 retail tasks."""
    source = DATA / "retail-tasks.json"
    if not source.exists():
        msg = f"{source} not found — run with --download first"
        raise SystemExit(msg)

    raw = json.loads(source.read_text(encoding="utf-8"))
    tasks: list[dict[str, Any]] = []

    for entry in raw:
        steps = reference_steps(entry)
        if not steps:
            continue
        tools_used = {step["tools"][0] for step in steps}
        if tools_used <= TRIVIAL_ONLY:
            continue

        task_id = f"tau2-retail-{str(entry['id']).zfill(3)}"
        # tau2 has no scenario grouping, so each task is its own cluster. That is the honest
        # encoding: these tasks are independent, and inventing clusters would understate the
        # effective sample size in the opposite direction from the usual mistake.
        tasks.append(
            {
                "id": task_id,
                "cluster_id": task_id,
                "template": "tau2_retail",
                "difficulty": "hard" if len(steps) > 3 else "medium",
                "tags": ["tau2", "retail", "tool_use"],
                "inputs": {"prompt": instruction_for(entry)},
                "checker": {"name": "trajectory_reference"},
                "reference": {
                    "trajectory": {
                        "required_tools": sorted(tools_used),
                        "allow_extra_calls": True,
                        "steps": steps,
                    }
                },
                "metadata": {"tau2_task_id": str(entry["id"]), "n_gold_actions": len(steps)},
            }
        )
        if limit and len(tasks) >= limit:
            break

    return {
        "schema_version": 1,
        "name": "tau2_retail",
        "version": "1.0.0",
        "description": (
            "Single-turn adaptation of the tau2-bench retail domain (MIT, "
            "sierra-research/tau2-bench). Tasks, tool surface and gold trajectories are theirs; "
            "the interaction protocol is ours. tau2-bench is multi-turn with a simulated user, "
            "so scores here are NOT comparable to a tau2-bench leaderboard."
        ),
        "default_k": 3,
        "agent": "tau2_retail_agent",
        "metadata": {
            "source": "https://github.com/sierra-research/tau2-bench",
            "source_license": "MIT",
            "adaptation": "single-turn; user scenario flattened, unknown_info withheld",
        },
        "tasks": tasks,
    }


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch tau2 data first")
    parser.add_argument("--limit", type=int, default=None, help="Only convert the first N tasks")
    args = parser.parse_args()

    if args.download:
        download()

    document = convert(limit=args.limit)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Generated by scripts/build_tau2_suite.py — do not edit by hand.\n"
        "#\n"
        "# Source: tau2-bench (MIT) — https://github.com/sierra-research/tau2-bench\n"
        "# Gold trajectories are tau2's own evaluation_criteria.actions.\n"
        "# This is a SINGLE-TURN adaptation; scores are not tau2-bench leaderboard scores.\n"
    )
    OUTPUT.write_text(
        header + yaml.safe_dump(document, sort_keys=False, width=100), encoding="utf-8"
    )
    n = len(document["tasks"])
    tools = {
        step["tools"][0]
        for t in document["tasks"]
        for step in t["reference"]["trajectory"]["steps"]
    }
    print(f"wrote {OUTPUT}: {n} tasks, {len(tools)} distinct gold tools")


if __name__ == "__main__":
    main()
