"""Shared fixtures and builders used across the test suites."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from agentgate.schemas import (
    BudgetSpec,
    CheckerSpec,
    Difficulty,
    FinalStep,
    LLMCallStep,
    ModelRef,
    ProviderMode,
    ReferenceSpec,
    ReferenceStep,
    ReferenceTrajectory,
    RunManifest,
    SuiteRef,
    SuiteSpec,
    TaskSpec,
    TokenUsage,
    ToolCallStep,
    ToolResultStep,
    Trajectory,
)

FIXED_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def make_task(
    task_id: str = "t1",
    *,
    cluster_id: str = "c1",
    prompt: str = "Refund order 42 for customer 7.",
    reference: ReferenceSpec | None = None,
) -> TaskSpec:
    """Build a small task with a two-step gold trajectory."""
    if reference is None:
        reference = ReferenceSpec(
            answer="Refunded order 42.",
            trajectory=ReferenceTrajectory(
                steps=[
                    ReferenceStep(tools=["lookup_customer"], args={"customer_id": 7}),
                    ReferenceStep(tools=["refund", "refund_order"], args={"order_id": 42}),
                ],
                required_tools=["refund"],
            ),
        )
    return TaskSpec(
        id=task_id,
        cluster_id=cluster_id,
        template="refund_flow",
        inputs={"prompt": prompt},
        reference=reference,
        checker=CheckerSpec(name="goal_state"),
        tags=["crm"],
        difficulty=Difficulty.MEDIUM,
    )


def make_suite(n_tasks: int = 4, *, clusters: int = 2, name: str = "smoke") -> SuiteSpec:
    """Build a suite with ``n_tasks`` tasks spread over ``clusters`` clusters."""
    tasks = [
        make_task(f"t{i}", cluster_id=f"c{i % clusters}", prompt=f"Task number {i}.")
        for i in range(n_tasks)
    ]
    return SuiteSpec(name=name, version="1.0.0", tasks=tasks, default_k=4)


def make_trajectory(
    task_id: str = "t1",
    *,
    rep: int = 0,
    system: str = "baseline",
    tools: tuple[str, ...] = ("lookup_customer", "refund"),
    args: tuple[dict[str, Any], ...] | None = None,
    answer: str = "Refunded order 42.",
    failing_call_index: int | None = None,
) -> Trajectory:
    """Build a trajectory that invokes ``tools`` in order and then answers."""
    if args is None:
        args = ({"customer_id": 7}, {"order_id": 42})[: len(tools)]
    traj = Trajectory(task_id=task_id, rep=rep, seed=1234 + rep, system=system, agent="tool_agent")
    for i, tool in enumerate(tools):
        traj.add_step(
            LLMCallStep(
                index=0,
                model="mock/agent",
                prompt_hash=f"h{i}",
                response_text=f"calling {tool}",
                usage=TokenUsage(prompt_tokens=100, completion_tokens=20),
            )
        )
        call_id = f"call-{i}"
        traj.add_step(
            ToolCallStep(
                index=0, call_id=call_id, tool=tool, args=dict(args[i]) if i < len(args) else {}
            )
        )
        ok = failing_call_index != i
        traj.add_step(
            ToolResultStep(
                index=0,
                call_id=call_id,
                tool=tool,
                ok=ok,
                output=f"{tool} ok" if ok else None,
                error=None if ok else "boom",
            )
        )
    traj.add_step(FinalStep(index=0, answer=answer))
    traj.final_answer = answer
    traj.latency_ms = 250.0 * len(tools)
    return traj


def make_manifest(
    run_id: str = "run-baseline",
    *,
    suite: SuiteSpec | None = None,
    system: str = "baseline",
    k: int = 4,
    base_seed: int = 20260101,
) -> RunManifest:
    """Build a manifest pinned to fixed time and model refs."""
    suite = suite or make_suite()
    return RunManifest(
        run_id=run_id,
        created_at=FIXED_TIME,
        agentgate_version="0.1.0.dev0",
        git_sha="0" * 40,
        suite=SuiteRef.from_suite(suite),
        system=system,
        agent="tool_agent",
        k=k,
        base_seed=base_seed,
        mode=ProviderMode.MOCK,
        models=[ModelRef(role="agent", model_id="mock/agent", provider="mock")],
        budget=BudgetSpec(max_requests=1000),
    )


@pytest.fixture
def suite() -> SuiteSpec:
    """A four-task, two-cluster suite."""
    return make_suite()


@pytest.fixture
def task() -> TaskSpec:
    """A single CRM refund task."""
    return make_task()


@pytest.fixture
def trajectory() -> Trajectory:
    """A successful two-tool trajectory."""
    return make_trajectory()
