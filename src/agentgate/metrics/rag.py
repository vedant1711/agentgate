"""RAG metrics — B3.

Native implementations, deliberately RAGAS-comparable (Es et al., arXiv:2309.15217) but not
RAGAS-dependent: the parity adapters in Phase 8 quantify where they agree and document where
they diverge. Two known divergences, stated here so they are never a surprise in the report:

* ``rag.answer_relevancy`` measures question-answer similarity directly, whereas RAGAS averages
  similarity between the question and LLM-generated questions reconstructed from the answer.
  Ours needs no generation step, which is what makes it free and deterministic.
* ``rag.context_precision`` judges a retrieved chunk against the task's declared gold contexts
  rather than asking a judge whether it *looks* relevant.

``rag.hallucination_rate`` is reported explicitly, as ``1 - faithfulness``, because stakeholders
ask for it by name (E1).
"""

from __future__ import annotations

from typing import ClassVar

from agentgate.metrics.base import BaseMetric, Scored, ScoredSample, ratio
from agentgate.metrics.lexical_judge import content_words
from agentgate.metrics.matching import cosine_of
from agentgate.metrics.registry import register
from agentgate.schemas.common import Direction, DType, MetricFamily, Requirement

RAG = MetricFamily.RAG

CONTEXT_RELEVANCE_THRESHOLD = 0.5
"""Content-word overlap above which a retrieved chunk counts as one of the gold contexts."""


def _faithfulness(sample: ScoredSample) -> tuple[float, dict[str, object]] | None:
    """Compute faithfulness, shared by the metric and its hallucination-rate complement."""
    judge = sample.context.judge
    contexts = sample.trajectory.retrieved_contexts
    if judge is None or not contexts:
        return None
    claims = judge.extract_claims(sample.answer)
    if not claims:
        return None
    supported = judge.check_claims(claims, contexts)
    value = ratio(sum(supported), len(claims))
    return value, {
        "judge": judge.name,
        "claims": claims,
        "supported": [bool(flag) for flag in supported],
        "n_contexts": len(contexts),
    }


@register
class Faithfulness(BaseMetric):
    """Claims in the answer supported by the retrieved contexts / total claims."""

    name = "rag.faithfulness"
    family: ClassVar[MetricFamily] = RAG
    dtype: ClassVar[DType] = "proportion"
    requires: ClassVar[set[Requirement]] = {Requirement.CONTEXTS, Requirement.JUDGE}
    description: ClassVar[str] = (
        "Judge-assisted claim decomposition of the answer, each claim checked against the "
        "retrieved contexts."
    )

    def compute(self, sample: ScoredSample) -> Scored:
        """Decompose the answer into claims and check each against the contexts."""
        result = _faithfulness(sample)
        if result is None:
            return Scored.skip("no factual claims in the answer")
        value, detail = result
        return Scored(value=value, detail=detail)


@register
class HallucinationRate(BaseMetric):
    """1 - faithfulness, reported explicitly because stakeholders ask for it by name (E1)."""

    name = "rag.hallucination_rate"
    family: ClassVar[MetricFamily] = RAG
    dtype: ClassVar[DType] = "proportion"
    direction: ClassVar[Direction] = "lower_is_better"
    requires: ClassVar[set[Requirement]] = {Requirement.CONTEXTS, Requirement.JUDGE}
    description: ClassVar[str] = "Share of answer claims unsupported by the retrieved contexts."

    def compute(self, sample: ScoredSample) -> Scored:
        """Complement of faithfulness."""
        result = _faithfulness(sample)
        if result is None:
            return Scored.skip("no factual claims in the answer")
        value, detail = result
        return Scored(value=1.0 - value, detail=detail)


@register
class AnswerRelevancy(BaseMetric):
    """How well the answer addresses the question that was actually asked."""

    name = "rag.answer_relevancy"
    family: ClassVar[MetricFamily] = RAG
    dtype: ClassVar[DType] = "proportion"
    requires: ClassVar[set[Requirement]] = {Requirement.EMBEDDINGS}
    description: ClassVar[str] = (
        "Embedding cosine between the question and the answer. An abstention scores 0: refusing "
        "to answer is sometimes correct, but it is never *relevant*, and outcome.abstained is "
        "where abstention is credited."
    )

    def compute(self, sample: ScoredSample) -> Scored:
        """Score question-answer similarity."""
        embedder = sample.context.embedder
        if embedder is None:  # pragma: no cover - guarded by `requires`
            return Scored.skip("no embedder configured")
        from agentgate.metrics.checkers import is_abstention

        if is_abstention(sample.answer):
            return Scored(value=0.0, detail={"abstained": True})
        similarity = cosine_of(sample.task.prompt, sample.answer, embedder)
        return Scored(value=similarity, detail={"embedder": embedder.name})


@register
class ContextPrecision(BaseMetric):
    """Proportion of retrieved chunks that are actually relevant."""

    name = "rag.context_precision"
    family: ClassVar[MetricFamily] = RAG
    dtype: ClassVar[DType] = "proportion"
    requires: ClassVar[set[Requirement]] = {Requirement.CONTEXTS}
    description: ClassVar[str] = (
        "Retrieved chunks matching one of the task's gold contexts / total retrieved chunks. "
        "Falls to zero when the retriever pads top-k with noise."
    )

    def compute(self, sample: ScoredSample) -> Scored:
        """Score retrieval precision against the declared gold contexts."""
        gold = sample.reference.contexts
        if not gold:
            return Scored.skip("task declares no gold contexts")
        retrieved = sample.trajectory.retrieved_contexts
        hits = [chunk for chunk in retrieved if _matches_any(chunk, gold)]
        return Scored(
            value=ratio(len(hits), len(retrieved)),
            detail={"retrieved": len(retrieved), "relevant": len(hits)},
        )


@register
class ContextRecall(BaseMetric):
    """Reference-answer claims attributable to the retrieved contexts."""

    name = "rag.context_recall"
    family: ClassVar[MetricFamily] = RAG
    dtype: ClassVar[DType] = "proportion"
    requires: ClassVar[set[Requirement]] = {Requirement.CONTEXTS, Requirement.REFERENCE_ANSWER}
    description: ClassVar[str] = (
        "Share of the reference answer's content that the retrieved contexts could have "
        "supported. This is the metric a context-budget cut moves first."
    )

    def compute(self, sample: ScoredSample) -> Scored:
        """Score how much of the reference answer the retrieval actually covered."""
        reference = sample.reference.answer
        if not reference:
            return Scored.skip("task declares no reference answer")
        retrieved = sample.trajectory.retrieved_contexts
        judge = sample.context.judge
        if judge is not None:
            claims = judge.extract_claims(reference) or [reference]
            supported = judge.check_claims(claims, retrieved)
            return Scored(
                value=ratio(sum(supported), len(claims)),
                detail={"judge": judge.name, "claims": claims},
            )
        words = content_words(reference)
        if not words:
            return Scored.skip("reference answer has no content words")
        covered = (
            words & set().union(*(content_words(chunk) for chunk in retrieved))
            if retrieved
            else set()
        )
        return Scored(
            value=ratio(len(covered), len(words)),
            detail={"covered": sorted(covered), "expected": sorted(words)},
        )


def _matches_any(chunk: str, gold: list[str]) -> bool:
    """Does ``chunk`` correspond to one of the gold contexts?

    Substring containment first (gold contexts are often exact excerpts), then content-word
    overlap for the chunked case where boundaries do not line up.
    """
    normalized = chunk.strip()
    for reference in gold:
        candidate = reference.strip()
        if not candidate:
            continue
        if candidate in normalized or normalized in candidate:
            return True
        expected = content_words(candidate)
        if expected and len(expected & content_words(normalized)) / len(expected) >= (
            CONTEXT_RELEVANCE_THRESHOLD
        ):
            return True
    return False
