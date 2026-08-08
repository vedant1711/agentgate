"""``tau2_retail_agent`` — a ReAct agent over the tau2-bench retail domain.

Unlike the three built-in reference agents, this one has **no deterministic brain**. It is only
meaningful driven by a real tool-calling model, because the whole point of adopting a published
benchmark is to stop measuring a specimen and start measuring a model.

Running it in ``mock`` mode is therefore refused rather than faked: a fabricated trajectory over
real benchmark tasks would produce numbers that look authoritative and mean nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from agentgate.agents.protocol import AgentConfig, BaseAgent, StepLimitError
from agentgate.agents.tau2_retail import DEFAULT_DB_PATH, RETAIL_POLICY, Tau2RetailSandbox
from agentgate.errors import ConfigError
from agentgate.providers.types import ChatMessage
from agentgate.schemas.task import TaskSpec
from agentgate.schemas.trajectory import FinalStep, Trajectory

MOCK_PREFIX = "mock/"


@dataclass
class Tau2RetailAgent(BaseAgent):
    """A tool-calling agent over the tau2-bench retail environment."""

    name: ClassVar[str] = "tau2_retail_agent"

    config: AgentConfig = field(
        default_factory=lambda: AgentConfig(model="ollama_chat/llama3.2:3b", max_steps=14)
    )
    db_path: Path = DEFAULT_DB_PATH

    def system_prompt(self, task: TaskSpec) -> str:  # noqa: ARG002 - policy is task-independent
        """Return the retail policy, minus any blocks the active faults removed."""
        prompt = RETAIL_POLICY
        if self.faults.prompt_degrade:
            prompt = _drop_block(prompt, "POLICY (must follow):")
        if self.faults.injection_vuln:
            prompt = _drop_block(prompt, "SECURITY (must follow):")
        if self.faults.verbosity:
            prompt += "\nSTYLE: Be exhaustive and restate context generously."
        return prompt

    async def execute(self, task: TaskSpec, seed: int, trajectory: Trajectory) -> str:
        """Run the ReAct loop against a fresh retail database."""
        if self.model.startswith(MOCK_PREFIX):
            msg = (
                f"{self.name} has no deterministic brain and must not be run in mock mode. "
                f"Fabricated trajectories over real benchmark tasks would look authoritative "
                f"and mean nothing. Pass --model with a real tool-calling model, e.g. "
                f"ollama_chat/llama3.2:3b."
            )
            raise ConfigError(msg)

        sandbox = Tau2RetailSandbox(db_path=self.db_path, faults=self.faults, seed=seed)
        try:
            return await self._loop(task, seed, trajectory, sandbox)
        finally:
            trajectory.sandbox_events = list(sandbox.events)
            trajectory.final_state = sandbox.snapshot()
            trajectory.metadata["available_tools"] = sandbox.tool_names
            trajectory.metadata["domain"] = "tau2-retail"
            sandbox.close()

    async def _loop(
        self, task: TaskSpec, seed: int, trajectory: Trajectory, sandbox: Tau2RetailSandbox
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


def _drop_block(prompt: str, marker: str) -> str:
    """Remove one labelled paragraph, the way a real 'prompt simplification' would."""
    lines = prompt.splitlines()
    kept: list[str] = []
    dropping = False
    for line in lines:
        if line.startswith(marker):
            dropping = True
            continue
        if dropping and (not line.strip() or not line.startswith("-")):
            dropping = False
        if not dropping:
            kept.append(line)
    return "\n".join(kept)
