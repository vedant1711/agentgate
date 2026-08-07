"""The integration seam: ``AgentUnderTest`` (A3.5).

An agent joins AgentGate by implementing one async method. The harness never reaches inside it,
never patches it, and never asks it to import anything from the metrics or gate layers — it
observes only the :class:`~agentgate.schemas.trajectory.Trajectory` that comes back.

:class:`BaseAgent` is a convenience for the three reference agents (and for anyone whose agent
is a tool-calling loop); it owns timing, error capture, fault plumbing, and tracing so a
subclass only has to describe *what the agent does*.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import ClassVar, Protocol, runtime_checkable

from agentgate.faults.config import FaultConfig
from agentgate.providers.client import LLMClient
from agentgate.providers.types import ChatMessage, ChatRequest, ChatResponse
from agentgate.schemas.task import TaskSpec
from agentgate.schemas.trajectory import (
    LLMCallStep,
    RunStatus,
    ToolCallStep,
    ToolResultStep,
    Trajectory,
)
from agentgate.tracing import (
    AGENTGATE_REP,
    AGENTGATE_SEED,
    AGENTGATE_SYSTEM,
    AGENTGATE_TASK_ID,
    KIND_AGENT,
    KIND_LLM,
    KIND_TOOL,
    LLM_COMPLETION_TOKENS,
    LLM_MODEL_NAME,
    LLM_PROMPT_TOKENS,
    LLM_TEMPERATURE,
    OUTPUT_VALUE,
    SPAN_KIND,
    TOOL_NAME,
    span,
)


@runtime_checkable
class AgentUnderTest(Protocol):
    """The whole integration contract.

    Implement ``name`` and ``run``; everything else in AgentGate is downstream of the returned
    trajectory. See ``examples/integrate_your_agent/`` for a 30-line example.
    """

    name: str

    async def run(self, task: TaskSpec, seed: int) -> Trajectory:
        """Execute ``task`` once and return the observable record of what happened."""
        ...


@dataclass(slots=True)
class AgentConfig:
    """Knobs a suite or run config may set on a reference agent."""

    model: str = "mock/agent"
    temperature: float = 0.0
    max_steps: int = 12
    top_k: int = 4
    system: str = "baseline"
    """The system-under-test label recorded on every trajectory (``baseline``/``candidate``)."""


@dataclass
class BaseAgent(ABC):
    """Timing, error handling, fault plumbing, and tracing for tool-calling agents.

    Args:
        client: Provider client. Its mode decides whether calls are live, cached, replayed, or
            mocked — the agent neither knows nor cares.
        config: Model and loop configuration.
        faults: Active regression knobs.
    """

    name: ClassVar[str] = "base"

    client: LLMClient
    config: AgentConfig = field(default_factory=AgentConfig)
    faults: FaultConfig = field(default_factory=FaultConfig)

    # -- fault-aware settings ---------------------------------------------

    @property
    def model(self) -> str:
        """The model actually called, honouring ``FAULT_MODEL_SWAP``."""
        return self.faults.model_swap or self.config.model

    @property
    def temperature(self) -> float:
        """The sampling temperature actually used, honouring ``FAULT_TEMPERATURE``."""
        return (
            self.config.temperature if self.faults.temperature is None else self.faults.temperature
        )

    @property
    def top_k(self) -> int:
        """Retrieval depth, honouring ``FAULT_TRUNCATE_CONTEXT``."""
        return max(1, round(self.config.top_k * self.faults.truncate_context))

    @abstractmethod
    def system_prompt(self, task: TaskSpec) -> str:
        """Return the system prompt for ``task``, with fault-driven blocks already removed."""

    @abstractmethod
    async def execute(self, task: TaskSpec, seed: int, trajectory: Trajectory) -> str:
        """Run the agent, appending steps to ``trajectory``.

        Returns:
            The final answer.
        """

    # -- the protocol method ----------------------------------------------

    async def run(self, task: TaskSpec, seed: int) -> Trajectory:
        """Execute ``task`` once under ``seed`` and return its trajectory.

        Errors are *recorded*, never raised: a crashed agent is a data point (a failed task),
        not a broken harness. Only the harness's own bugs propagate.
        """
        trajectory = Trajectory(
            task_id=task.id,
            rep=0,
            seed=seed,
            system=self.config.system,
            agent=self.name,
            faults=self.faults.active(),
            started_at=datetime.now(UTC),
        )
        started = time.perf_counter()
        attributes = {
            SPAN_KIND: KIND_AGENT,
            AGENTGATE_TASK_ID: task.id,
            AGENTGATE_SEED: seed,
            AGENTGATE_SYSTEM: self.config.system,
            AGENTGATE_REP: trajectory.rep,
        }
        with span(f"agent.{self.name}", attributes) as agent_span:
            try:
                trajectory.final_answer = await self.execute(task, seed, trajectory)
            except StepLimitError:
                trajectory.status = RunStatus.MAX_STEPS
                trajectory.error = f"exceeded {self.config.max_steps} steps without answering"
            except Exception as exc:
                trajectory.status = RunStatus.ERROR
                trajectory.error = f"{type(exc).__name__}: {exc}"
                agent_span.record_error(trajectory.error)
            agent_span.set_attribute(OUTPUT_VALUE, trajectory.final_answer[:2000])

        trajectory.latency_ms = (time.perf_counter() - started) * 1000.0
        trajectory.ended_at = datetime.now(UTC)
        return trajectory

    # -- helpers for subclasses -------------------------------------------

    async def call_model(
        self,
        messages: list[ChatMessage],
        trajectory: Trajectory,
        *,
        seed: int,
        tools: list[object] | None = None,
        reasoning: str | None = None,
    ) -> ChatResponse:
        """Issue one completion, recording an ``llm_call`` step and a span.

        Args:
            messages: Conversation so far.
            trajectory: Trajectory to append the step to.
            seed: Deterministic seed for this (task, rep).
            tools: Tool specs to expose, if any.
            reasoning: Intermediate reasoning to record for grounded-reasoning scoring.

        Returns:
            The provider response.
        """
        request = ChatRequest(
            model=self.model,
            messages=messages,
            tools=tools or [],  # type: ignore[arg-type]
            temperature=self.temperature,
            seed=seed,
        )
        attributes = {
            SPAN_KIND: KIND_LLM,
            LLM_MODEL_NAME: self.model,
            LLM_TEMPERATURE: self.temperature,
        }
        with span("llm.completion", attributes) as llm_span:
            response = await self.client.complete(request)
            llm_span.set_attribute(LLM_PROMPT_TOKENS, response.usage.prompt_tokens)
            llm_span.set_attribute(LLM_COMPLETION_TOKENS, response.usage.completion_tokens)
            span_id = llm_span.span_id
        trajectory.add_step(
            LLMCallStep(
                index=0,
                model=self.model,
                prompt_hash=request.prompt_hash(),
                response_text=response.text,
                usage=response.usage,
                cached=response.cached,
                temperature=self.temperature,
                finish_reason=response.finish_reason,
                reasoning=reasoning,
                duration_ms=response.latency_ms,
                span_id=span_id,
            )
        )
        return response

    def record_tool(
        self,
        trajectory: Trajectory,
        *,
        call_id: str,
        tool: str,
        args: dict[str, object],
        ok: bool,
        output: object = None,
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> ChatMessage:
        """Record a tool call and its result, and return the message to feed back to the model.

        Results are prefixed ``OK:``/``ERROR:`` so a brain (and a human reading the trajectory)
        can tell success from failure without parsing the payload.
        """
        with span("tool.execute", {SPAN_KIND: KIND_TOOL, TOOL_NAME: tool}) as tool_span:
            if not ok and error:
                tool_span.record_error(error)
            span_id = tool_span.span_id
        trajectory.add_step(
            ToolCallStep(index=0, call_id=call_id, tool=tool, args=dict(args), span_id=span_id)
        )
        trajectory.add_step(
            ToolResultStep(
                index=0,
                call_id=call_id,
                tool=tool,
                ok=ok,
                output=output,
                error=error,
                duration_ms=duration_ms,
                span_id=span_id,
            )
        )
        body = f"OK: {json.dumps(output, default=str)}" if ok else f"ERROR: {error}"
        return ChatMessage(role="tool", content=body, name=tool, tool_call_id=call_id)


class StepLimitError(RuntimeError):
    """Raised by an agent loop that hit ``max_steps`` without producing an answer."""
