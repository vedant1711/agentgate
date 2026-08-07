"""``rag_agent`` — retrieval-augmented QA over the local wiki corpus (F1.2).

Retrieval happens inside the agent (recorded as ``retrieved_contexts``) rather than as a tool
call, because this agent exists to exercise the RAG metric family: faithfulness, answer
relevancy, context precision/recall. ``FAULT_TRUNCATE_CONTEXT`` cuts top-k here, which is the
realistic shape of a "we reduced the context budget to save tokens" regression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from agentgate.agents.corpus import WikiCorpus
from agentgate.agents.prompts import rag_agent_prompt
from agentgate.agents.protocol import AgentConfig, BaseAgent
from agentgate.agents.retrieval import BM25Index
from agentgate.providers.types import ChatMessage
from agentgate.schemas.task import TaskSpec
from agentgate.schemas.trajectory import FinalStep, Trajectory
from agentgate.tracing import KIND_RETRIEVER, RETRIEVAL_DOCUMENTS, SPAN_KIND, span

CONTEXT_HEADER = "Context:"


@dataclass
class RagAgent(BaseAgent):
    """Answers wiki questions from retrieved passages, abstaining when they are missing."""

    name: ClassVar[str] = "rag_agent"

    config: AgentConfig = field(default_factory=lambda: AgentConfig(max_steps=1, top_k=4))
    corpus: WikiCorpus = field(default_factory=WikiCorpus)
    _index: BM25Index | None = field(default=None, init=False, repr=False)

    @property
    def index(self) -> BM25Index:
        """The BM25 index, built once per agent instance."""
        if self._index is None:
            self._index = BM25Index([(doc.doc_id, doc.text) for doc in self.corpus.docs])
        return self._index

    def system_prompt(self, task: TaskSpec) -> str:  # noqa: ARG002 - prompt is task-independent
        """Return the RAG system prompt, minus any blocks the active faults removed."""
        return rag_agent_prompt(self.faults)

    async def execute(self, task: TaskSpec, seed: int, trajectory: Trajectory) -> str:
        """Retrieve, then answer from the retrieved passages only."""
        question = task.prompt
        with span("retriever.search", {SPAN_KIND: KIND_RETRIEVER}) as retriever_span:
            hits = self.index.search(question, top_k=self.top_k)
            retriever_span.set_attribute(RETRIEVAL_DOCUMENTS, len(hits))
        contexts = [self._budget(hit.text) for hit in hits]
        trajectory.retrieved_contexts = contexts
        trajectory.metadata["retrieved_doc_ids"] = [hit.doc_id for hit in hits]
        trajectory.metadata["top_k"] = self.top_k

        context = "\n\n".join(
            f"[{hit.doc_id}]\n{text}" for hit, text in zip(hits, contexts, strict=True)
        )
        messages = [
            ChatMessage(role="system", content=self.system_prompt(task)),
            ChatMessage(role="user", content=f"{question}\n\n{CONTEXT_HEADER}\n{context}"),
        ]
        response = await self.call_model(messages, trajectory, seed=seed)
        trajectory.add_step(FinalStep(index=0, answer=response.text))
        return response.text

    def _budget(self, text: str) -> str:
        """Apply the context-window budget to one passage.

        ``FAULT_TRUNCATE_CONTEXT`` cuts *both* top-k and per-passage length, because that is what
        a real "we reduced the context budget" change does — and cutting only top-k would leave
        this corpus unharmed, since the gold document usually ranks first.
        """
        if self.faults.truncate_context >= 1.0:
            return text
        keep = max(1, int(len(text) * self.faults.truncate_context))
        return text[:keep]
