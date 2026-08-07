"""BM25 retrieval over the local corpus.

Deliberately dependency-free and deterministic. An embedding index would be more accurate, but
retrieval quality is not what AgentGate is measuring — retrieval *stability* is, because the
``FAULT_TRUNCATE_CONTEXT`` knob needs a top-k that means exactly what it says on every machine.
The optional ``embeddings`` extra can swap in a local sentence-transformers index without
changing this interface.

Formula: Okapi BM25 (Robertson & Zaragoza, "The Probabilistic Relevance Framework"), with the
conventional ``k1=1.5`` and ``b=0.75``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "within",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on word characters, and drop stopwords and single characters."""
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    ]


@dataclass(frozen=True, slots=True)
class Retrieved:
    """One retrieval hit."""

    doc_id: str
    text: str
    score: float
    rank: int


class BM25Index:
    """An Okapi BM25 index over ``(doc_id, text)`` pairs.

    Args:
        documents: Corpus entries.
        k1: Term-frequency saturation parameter.
        b: Length-normalisation parameter.
    """

    def __init__(
        self, documents: Sequence[tuple[str, str]], *, k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.k1 = k1
        self.b = b
        self._doc_ids: list[str] = [doc_id for doc_id, _ in documents]
        self._texts: list[str] = [text for _, text in documents]
        self._tokens: list[Counter[str]] = [Counter(tokenize(text)) for text in self._texts]
        self._lengths: list[int] = [sum(counts.values()) for counts in self._tokens]
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._doc_freq: Counter[str] = Counter()
        for counts in self._tokens:
            self._doc_freq.update(counts.keys())

    def __len__(self) -> int:
        """Number of indexed documents."""
        return len(self._doc_ids)

    def _idf(self, term: str) -> float:
        """Robertson-Sparck-Jones IDF with the +0.5 smoothing that keeps it non-negative."""
        n = len(self._doc_ids)
        df = self._doc_freq.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str, index: int) -> float:
        """Return the BM25 score of document ``index`` for ``query``."""
        counts = self._tokens[index]
        length = self._lengths[index] or 1
        total = 0.0
        for term in tokenize(query):
            freq = counts.get(term, 0)
            if not freq:
                continue
            denominator = freq + self.k1 * (1.0 - self.b + self.b * length / (self._avg_len or 1.0))
            total += self._idf(term) * (freq * (self.k1 + 1.0)) / denominator
        return total

    def search(self, query: str, top_k: int = 4) -> list[Retrieved]:
        """Return the ``top_k`` highest-scoring documents.

        Ties break on document id so results are stable across runs — a retriever that
        reorders equal-scoring hits would inject noise into every RAG metric.

        Args:
            query: Free-text query.
            top_k: Maximum hits to return.

        Returns:
            Hits in descending score order, with zero-scoring documents excluded.
        """
        scored = [
            (self.score(query, index), self._doc_ids[index], self._texts[index])
            for index in range(len(self._doc_ids))
        ]
        scored = [entry for entry in scored if entry[0] > 0.0]
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        return [
            Retrieved(doc_id=doc_id, text=text, score=score, rank=rank)
            for rank, (score, doc_id, text) in enumerate(scored[: max(0, top_k)])
        ]
