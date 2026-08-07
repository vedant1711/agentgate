"""``tool_agent`` — a ReAct-style tool-calling agent over the mock enterprise sandbox (F1.1).

Success is judged tau-bench style against the sandbox's **end state** (E2), not against the
wording of the answer: whether the refund actually happened is a fact about the database, and a
fluent apology that changed nothing must not score as a success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from agentgate.agents.prompts import tool_agent_prompt
from agentgate.agents.protocol import AgentConfig, BaseAgent, StepLimitError
from agentgate.agents.sandbox import Sandbox
from agentgate.providers.types import ChatMessage
from agentgate.schemas.task import TaskSpec
from agentgate.schemas.trajectory import FinalStep, Trajectory
from agentgate.seeds import derive_seed


@dataclass
class ToolAgent(BaseAgent):
    """A tool-calling agent for multi-step CRM and operations workflows."""

    name: ClassVar[str] = "tool_agent"

    config: AgentConfig = field(default_factory=lambda: AgentConfig(max_steps=12))

    def system_prompt(self, task: TaskSpec) -> str:  # noqa: ARG002 - prompt is task-independent
        """Return the ops system prompt, minus any blocks the active faults removed."""
        return tool_agent_prompt(self.faults)

    async def execute(self, task: TaskSpec, seed: int, trajectory: Trajectory) -> str:
        """Run the ReAct loop against a fresh sandbox and return the final answer."""
        # Mix the task id into the sandbox seed so a flaky-dependency simulation injects a
        # *different* failure pattern per task. One shared stream would make every task fail in
        # the same place, which is not what flakiness looks like.
        sandbox = Sandbox.for_task(task, faults=self.faults, seed=derive_seed(seed, task.id))
        try:
            return await self._loop(task, seed, trajectory, sandbox)
        finally:
            trajectory.sandbox_events = list(sandbox.events)
            trajectory.final_state = sandbox.snapshot()
            trajectory.metadata["available_tools"] = sandbox.tool_names
            sandbox.close()

    async def _loop(
        self, task: TaskSpec, seed: int, trajectory: Trajectory, sandbox: Sandbox
    ) -> str:
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt(task)),
            ChatMessage(role="user", content=task.prompt),
        ]
        tools = sandbox.tools()
        max_steps = min(self.config.max_steps, task.max_steps)

        for _ in range(max_steps):
            response = await self.call_model(messages, trajectory, seed=seed, tools=list(tools))
            if not response.tool_calls:
                trajectory.add_step(FinalStep(index=0, answer=response.text))
                return response.text

            messages.append(
                ChatMessage(role="assistant", content=response.text, tool_calls=response.tool_calls)
            )
            for call in response.tool_calls:
                outcome = sandbox.invoke(call.name, call.arguments)
                messages.append(
                    self.record_tool(
                        trajectory,
                        call_id=call.id,
                        tool=call.name,
                        args=call.arguments,
                        ok=outcome.ok,
                        output=outcome.output,
                        error=outcome.error,
                        duration_ms=outcome.latency_ms,
                    )
                )
        raise StepLimitError
