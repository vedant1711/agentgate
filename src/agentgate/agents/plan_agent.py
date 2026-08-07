"""``plan_agent`` — a two-stage planner/executor for longer-horizon research (F1.3).

Retrieval is a *tool call* here, unlike in ``rag_agent``, so this agent exercises the trajectory
family: step efficiency, redundant calls, loop detection, and grounded reasoning. Sampling drift
makes it re-run searches it has already run, which is exactly the failure
``trajectory.redundant_call_rate`` exists to surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from agentgate.agents.corpus import WikiCorpus
from agentgate.agents.prompts import plan_agent_prompt
from agentgate.agents.protocol import AgentConfig, BaseAgent, StepLimitError
from agentgate.agents.retrieval import BM25Index
from agentgate.providers.types import ChatMessage, ToolSpec
from agentgate.schemas.task import TaskSpec
from agentgate.schemas.trajectory import FinalStep, Trajectory

SEARCH_DOCS = ToolSpec(
    name="search_docs",
    description="Search the internal wiki. Returns document ids with a snippet.",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
)
READ_DOC = ToolSpec(
    name="read_doc",
    description="Read the full text of a document by id.",
    parameters={"type": "object", "properties": {"doc_id": {"type": "string"}}},
)


@dataclass
class PlanAgent(BaseAgent):
    """Plans, researches with tools, then summarises with citations."""

    name: ClassVar[str] = "plan_agent"

    config: AgentConfig = field(default_factory=lambda: AgentConfig(max_steps=8, top_k=3))
    corpus: WikiCorpus = field(default_factory=WikiCorpus)
    _index: BM25Index | None = field(default=None, init=False, repr=False)

    @property
    def index(self) -> BM25Index:
        """The BM25 index, built once per agent instance."""
        if self._index is None:
            self._index = BM25Index([(doc.doc_id, doc.text) for doc in self.corpus.docs])
        return self._index

    def system_prompt(self, task: TaskSpec) -> str:  # noqa: ARG002 - prompt is task-independent
        """Return the planner system prompt, minus any blocks the active faults removed."""
        return plan_agent_prompt(self.faults)

    def tools(self) -> list[ToolSpec]:
        """Tools visible to the planner, minus anything ``FAULT_DROP_TOOL`` removed."""
        return [spec for spec in (SEARCH_DOCS, READ_DOC) if not self.faults.hides_tool(spec.name)]

    async def execute(self, task: TaskSpec, seed: int, trajectory: Trajectory) -> str:
        """Run the plan/execute loop and return the cited summary."""
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self.system_prompt(task)),
            ChatMessage(role="user", content=task.prompt),
        ]
        tools = self.tools()
        max_steps = min(self.config.max_steps, task.max_steps)
        contexts: list[str] = []

        for step in range(max_steps):
            plan_note = f"Plan step {step + 1}: gather evidence, then summarise with citations."
            response = await self.call_model(
                messages, trajectory, seed=seed, tools=list(tools), reasoning=plan_note
            )
            if not response.tool_calls:
                trajectory.retrieved_contexts = contexts
                trajectory.add_step(FinalStep(index=0, answer=response.text))
                return response.text

            messages.append(
                ChatMessage(role="assistant", content=response.text, tool_calls=response.tool_calls)
            )
            for call in response.tool_calls:
                ok, output, error = self._invoke(call.name, call.arguments)
                if ok and call.name == "read_doc" and isinstance(output, dict):
                    contexts.append(str(output.get("text", "")))
                messages.append(
                    self.record_tool(
                        trajectory,
                        call_id=call.id,
                        tool=call.name,
                        args=call.arguments,
                        ok=ok,
                        output=output,
                        error=error,
                    )
                )
        trajectory.retrieved_contexts = contexts
        raise StepLimitError

    def _invoke(self, tool: str, args: dict[str, Any]) -> tuple[bool, Any, str | None]:
        """Execute a research tool against the corpus."""
        if self.faults.hides_tool(tool):
            return False, None, f"unknown tool {tool!r}"
        if tool == "search_docs":
            hits = self.index.search(str(args.get("query", "")), top_k=self.top_k)
            return (
                True,
                [
                    {"doc_id": hit.doc_id, "snippet": hit.text[:160], "score": round(hit.score, 4)}
                    for hit in hits
                ],
                None,
            )
        if tool == "read_doc":
            doc = self.corpus.get(str(args.get("doc_id", "")))
            if doc is None:
                return False, None, "document not found"
            return True, {"doc_id": doc.doc_id, "title": doc.title, "text": doc.text}, None
        return False, None, f"unknown tool {tool!r}"
