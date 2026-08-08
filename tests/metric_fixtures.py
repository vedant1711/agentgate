"""Loader for the committed golden-value metric fixtures.

Fixtures are YAML so the expected values can be *hand-computed and reviewed* rather than
captured from the implementation — a golden file regenerated from the code under test proves
only that the code is self-consistent.

The compact schema below builds real ``TaskSpec``/``Trajectory`` objects; see
``tests/fixtures/metrics/*.yaml`` for the cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agentgate.metrics.base import MetricContext, ScoredSample
from agentgate.schemas.task import (
    CheckerSpec,
    InjectionProbe,
    ReferenceSpec,
    ReferenceStep,
    ReferenceTrajectory,
    TaskSpec,
)
from agentgate.schemas.trajectory import (
    FinalStep,
    LLMCallStep,
    RunStatus,
    SandboxEvent,
    TokenUsage,
    ToolCallStep,
    ToolResultStep,
    Trajectory,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "metrics"


@dataclass(frozen=True, slots=True)
class GoldenCase:
    """One hand-computed expectation for one metric."""

    metric: str
    name: str
    expected: float | None
    status: str
    task: TaskSpec
    trajectory: Trajectory
    note: str = ""

    @property
    def label(self) -> str:
        """Readable id for pytest parametrisation."""
        return f"{self.metric}::{self.name}"


def _reference(spec: dict[str, Any]) -> ReferenceSpec:
    payload = dict(spec)
    trajectory = payload.pop("trajectory", None)
    injection = payload.pop("injection", None)
    reference = ReferenceSpec(
        **payload,
        trajectory=_reference_trajectory(trajectory) if trajectory else None,
        injection=InjectionProbe(**injection) if injection else None,
    )
    return reference


def _reference_trajectory(spec: dict[str, Any]) -> ReferenceTrajectory:
    steps = [
        ReferenceStep(
            tools=[step["tool"]] if "tool" in step else list(step["tools"]),
            args=step.get("args"),
            arg_comparators=step.get("arg_comparators", {}),
            optional=step.get("optional", False),
            group=step.get("group"),
        )
        for step in spec.get("steps", [])
    ]
    return ReferenceTrajectory(
        steps=steps,
        allow_extra_calls=spec.get("allow_extra_calls", True),
        required_tools=spec.get("required_tools", []),
    )


def _trajectory(spec: dict[str, Any], *, task_id: str) -> Trajectory:
    trajectory = Trajectory(
        task_id=task_id,
        rep=0,
        seed=1,
        system="baseline",
        agent="fixture",
        status=RunStatus(spec.get("status", "completed")),
        final_answer=spec.get("answer", ""),
        retrieved_contexts=list(spec.get("retrieved_contexts", [])),
        final_state=spec.get("final_state"),
        est_cost_usd=float(spec.get("est_cost_usd", 0.0)),
        sandbox_events=[SandboxEvent(**event) for event in spec.get("sandbox_events", [])],
    )
    usage = spec.get("usage", {})
    for reasoning in spec.get("reasoning", []) or [None]:
        if reasoning is None and not spec.get("reasoning"):
            break
        trajectory.add_step(
            LLMCallStep(index=0, model="fixture", prompt_hash="h", reasoning=reasoning)
        )
    for index, call in enumerate(_calls(spec)):
        call_id = f"c{index}"
        trajectory.add_step(
            ToolCallStep(index=0, call_id=call_id, tool=call["tool"], args=call.get("args", {}))
        )
        trajectory.add_step(
            ToolResultStep(
                index=0,
                call_id=call_id,
                tool=call["tool"],
                ok=call.get("ok", True),
                output=call.get("output"),
                error=None if call.get("ok", True) else call.get("error", "failed"),
                duration_ms=float(call.get("duration_ms", 0.0)),
            )
        )
    trajectory.add_step(FinalStep(index=0, answer=trajectory.final_answer))
    if usage:
        trajectory.usage = TokenUsage(**usage)
    trajectory.latency_ms = float(
        spec.get("latency_ms", sum(step.duration_ms for step in trajectory.steps))
    )
    return trajectory


def _calls(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if "calls" in spec:
        return [dict(call) for call in spec["calls"]]
    return [{"tool": tool} for tool in spec.get("tools", [])]


def load_cases(path: Path) -> list[GoldenCase]:
    """Load every case from one fixture file.

    A file holds either a single ``metric``/``cases`` pair or a list of ``groups``, so related
    metrics can share a file without repeating their scenarios.
    """
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    groups = document.get("groups") or [document]
    cases: list[GoldenCase] = []
    for group in groups:
        cases.extend(_load_group(group))
    return cases


def _load_group(document: dict[str, Any]) -> list[GoldenCase]:
    metric = document["metric"]
    cases: list[GoldenCase] = []
    for index, case in enumerate(document["cases"]):
        task_spec = case.get("task", {})
        task = TaskSpec(
            id=f"fx{index}",
            cluster_id="fixture",
            inputs={"prompt": task_spec.get("prompt", "do the thing")},
            reference=_reference(task_spec.get("reference", {})),
            checker=CheckerSpec(**task_spec["checker"]) if "checker" in task_spec else None,
        )
        cases.append(
            GoldenCase(
                metric=metric,
                name=case["name"],
                expected=case.get("expected"),
                status=case.get("status", "ok" if case.get("expected") is not None else "skipped"),
                task=task,
                trajectory=_trajectory(case.get("trajectory", {}), task_id=task.id),
                note=case.get("note", ""),
            )
        )
    return cases


def load_all_cases(directory: Path = FIXTURE_DIR) -> list[GoldenCase]:
    """Load every committed golden case, sorted by metric then case name."""
    cases: list[GoldenCase] = []
    for path in sorted(directory.glob("*.yaml")):
        cases.extend(load_cases(path))
    return cases


def sample_for(case: GoldenCase, context: MetricContext) -> ScoredSample:
    """Build the scored sample a golden case describes."""
    return ScoredSample(task=case.task, trajectory=case.trajectory, run_id="fx", context=context)
