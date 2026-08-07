"""Trajectory schema: the observable record of one agent execution.

A trajectory is the unit the metrics engine scores. It is deliberately a *record*, never a
live handle onto the agent — constraint A3.5: the harness never mutates the system under test,
it only observes through this structure.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import Field

from agentgate.schemas.common import AgentGateModel, FrozenModel


class RunStatus(StrEnum):
    """Terminal state of a single (task, repetition) execution."""

    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"
    MAX_STEPS = "max_steps"
    BUDGET_EXHAUSTED = "budget_exhausted"


class TokenUsage(AgentGateModel):
    """Token accounting for a step or a whole trajectory."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        """Sum of prompt and completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Add two usage records component-wise."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class _StepBase(AgentGateModel):
    """Fields common to every trajectory step."""

    index: int = Field(ge=0, description="Zero-based position in the step list.")
    duration_ms: float = Field(default=0.0, ge=0.0)
    span_id: str | None = Field(default=None, description="OpenTelemetry span id, when tracing.")


class LLMCallStep(_StepBase):
    """One round-trip to the agent's model."""

    kind: Literal["llm_call"] = "llm_call"
    model: str
    prompt_hash: str = Field(description="Hash of the exact messages sent; part of the cache key.")
    response_text: str = ""
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cached: bool = False
    temperature: float = 0.0
    finish_reason: str | None = None
    reasoning: str | None = Field(
        default=None,
        description="Intermediate reasoning text, scored by trajectory.grounded_reasoning.",
    )


class ToolCallStep(_StepBase):
    """The agent's request to invoke a tool."""

    kind: Literal["tool_call"] = "tool_call"
    call_id: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResultStep(_StepBase):
    """The sandbox's response to a tool call."""

    kind: Literal["tool_result"] = "tool_result"
    call_id: str
    tool: str
    ok: bool = True
    output: Any = None
    error: str | None = None


class FinalStep(_StepBase):
    """The agent's terminal answer."""

    kind: Literal["final"] = "final"
    answer: str = ""


Step = Annotated[
    LLMCallStep | ToolCallStep | ToolResultStep | FinalStep,
    Field(discriminator="kind"),
]
"""Discriminated union of trajectory steps."""


class SandboxEvent(FrozenModel):
    """A side effect the sandbox recorded, used by goal-state and safety checkers."""

    kind: str = Field(description="e.g. 'db_write', 'email_sent', 'file_written', 'refund'.")
    target: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    destructive: bool = False
    confirmed: bool = Field(
        default=False, description="Did the agent seek confirmation before a destructive action?"
    )


class ToolInvocation(FrozenModel):
    """A tool call paired with its result — the view trajectory metrics operate on."""

    index: int
    call_id: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    error: str | None = None

    @property
    def signature(self) -> str:
        """Stable ``tool(args)`` identity used for duplicate/loop detection (B2)."""
        from agentgate.schemas.common import canonical_dumps

        return f"{self.tool}({canonical_dumps(self.args)})"


class Trajectory(AgentGateModel):
    """The complete observable record of one (task, repetition) execution."""

    task_id: str
    rep: int = Field(ge=0, description="Repetition index in [0, K).")
    seed: int = Field(description="Deterministic seed derived from (run, task, rep).")
    system: str = Field(description="System-under-test label, e.g. 'baseline' or 'candidate'.")
    agent: str = Field(default="", description="Reference-agent id that produced this run.")

    steps: list[Step] = Field(default_factory=list)
    final_answer: str = ""
    status: RunStatus = RunStatus.COMPLETED
    error: str | None = None

    started_at: datetime | None = None
    ended_at: datetime | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)

    usage: TokenUsage = Field(default_factory=TokenUsage)
    est_cost_usd: float = Field(default=0.0, ge=0.0)

    retrieved_contexts: list[str] = Field(
        default_factory=list, description="Chunks the agent retrieved; input to the RAG family."
    )
    sandbox_events: list[SandboxEvent] = Field(default_factory=list)
    final_state: dict[str, Any] | None = Field(
        default=None, description="Sandbox end state compared against the task's goal_state."
    )
    faults: list[str] = Field(
        default_factory=list, description="Fault-injection knobs active for this execution."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- derived views -----------------------------------------------------

    @property
    def tool_invocations(self) -> list[ToolInvocation]:
        """Pair each ``tool_call`` with its matching ``tool_result``.

        Calls with no result (agent crashed mid-step) are reported as failed invocations so
        efficiency and error-recovery metrics still see them.
        """
        results: dict[str, ToolResultStep] = {
            step.call_id: step for step in self.steps if isinstance(step, ToolResultStep)
        }
        invocations: list[ToolInvocation] = []
        for step in self.steps:
            if not isinstance(step, ToolCallStep):
                continue
            result = results.get(step.call_id)
            invocations.append(
                ToolInvocation(
                    index=len(invocations),
                    call_id=step.call_id,
                    tool=step.tool,
                    args=step.args,
                    ok=result.ok if result is not None else False,
                    error=result.error if result is not None else "no tool result recorded",
                )
            )
        return invocations

    @property
    def tool_sequence(self) -> list[str]:
        """Ordered tool names, the input to sequence-match metrics."""
        return [call.tool for call in self.tool_invocations]

    @property
    def llm_calls(self) -> list[LLMCallStep]:
        """Every model round-trip in order."""
        return [step for step in self.steps if isinstance(step, LLMCallStep)]

    @property
    def n_tool_calls(self) -> int:
        """Count of tool invocations (``efficiency.tool_calls_count``)."""
        return sum(1 for step in self.steps if isinstance(step, ToolCallStep))

    @property
    def n_llm_roundtrips(self) -> int:
        """Count of model round-trips (``efficiency.llm_roundtrips``)."""
        return len(self.llm_calls)

    @property
    def reasoning_texts(self) -> list[str]:
        """Intermediate reasoning strings for grounded-reasoning scoring."""
        return [call.reasoning for call in self.llm_calls if call.reasoning]

    @property
    def evidence_bank(self) -> list[str]:
        """Serialized tool outputs available as evidence at any point in the trajectory."""
        bank: list[str] = list(self.retrieved_contexts)
        for step in self.steps:
            if isinstance(step, ToolResultStep) and step.ok and step.output is not None:
                bank.append(step.output if isinstance(step.output, str) else repr(step.output))
        return bank

    def add_step(self, step: LLMCallStep | ToolCallStep | ToolResultStep | FinalStep) -> None:
        """Append a step, assigning its index and folding in token usage."""
        step.index = len(self.steps)
        self.steps.append(step)
        if isinstance(step, LLMCallStep):
            self.usage = self.usage + step.usage
