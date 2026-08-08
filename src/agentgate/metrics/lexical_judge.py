"""A deterministic, non-LLM judge.

This exists so judge-backed metrics can be developed, unit-tested, and demonstrated offline for
free. It is **not** an LLM judge and never pretends to be: it splits text into sentences and
scores support by lexical overlap with the evidence. Every report names the judge that produced
each number, so a `lexical` number is never mistaken for a rubric-judged one.

Its real jobs:

* the default when no judge is configured, so a suite still gets *some* faithfulness signal;
* the fixture judge for tests, where determinism matters more than nuance;
* the control in Part D's bias experiments — a judge with, by construction, no position bias,
  no verbosity bias, and no self-preference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agentgate.metrics.base import JudgeVerdict
from agentgate.metrics.checkers import normalize_answer

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "within",
    ]
)

DEFAULT_SUPPORT_THRESHOLD = 0.6


def split_claims(text: str) -> list[str]:
    """Split text into atomic claims: non-trivial sentences.

    Args:
        text: Free text.

    Returns:
        Sentences with at least two content words, in order.
    """
    claims: list[str] = []
    for fragment in _SENTENCE_SPLIT.split(text or ""):
        candidate = fragment.strip()
        if candidate and len(content_words(candidate)) >= 2:
            claims.append(candidate)
    return claims


def content_words(text: str) -> set[str]:
    """Normalised content words, stopwords removed."""
    return {word for word in normalize_answer(text).split() if word not in _STOPWORDS}


def support_score(claim: str, evidence: list[str]) -> float:
    """Best fraction of a claim's content words found in any single evidence item.

    Args:
        claim: The claim to check.
        evidence: Candidate supporting texts.

    Returns:
        A value in [0, 1]; 0 when the claim has no content words or no evidence exists.
    """
    words = content_words(claim)
    if not words:
        return 0.0
    best = 0.0
    for item in evidence:
        overlap = len(words & content_words(item)) / len(words)
        best = max(best, overlap)
    return best


@dataclass(slots=True)
class LexicalJudge:
    """A deterministic judge backed by lexical overlap.

    Args:
        support_threshold: Overlap fraction above which a claim counts as supported.
    """

    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD
    name: str = "lexical"

    def extract_claims(self, text: str) -> list[str]:
        """Split ``text`` into atomic claims."""
        return split_claims(text)

    def check_claims(self, claims: list[str], evidence: list[str]) -> list[bool]:
        """Return, per claim, whether the evidence supports it."""
        return [support_score(claim, evidence) >= self.support_threshold for claim in claims]

    def score_criterion(
        self,
        criterion: str,
        prompt: str,
        response: str,
        *,
        reference: str = "",
        contexts: list[str] | None = None,
    ) -> JudgeVerdict:
        """Score one criterion deterministically.

        Every criterion reduces to overlap with something: the reference answer for
        correctness and completeness, the prompt for instruction-following, and internal
        repetition for coherence. Crude by design — the real judges arrive in Part D.
        """
        if criterion in ("correctness", "completeness") and reference:
            overlap = (
                support_score(reference, [response])
                if criterion == "completeness"
                else (support_score(response, [reference]))
            )
            return JudgeVerdict(value=overlap, samples=(overlap,), variance=0.0)
        if criterion == "instruction_following":
            overlap = support_score(prompt, [response])
            return JudgeVerdict(value=min(1.0, overlap * 2.0), samples=(overlap,), variance=0.0)
        if criterion == "faithfulness" and contexts:
            claims = self.extract_claims(response)
            if not claims:
                return JudgeVerdict(value=1.0, samples=(1.0,), variance=0.0)
            supported = sum(self.check_claims(claims, contexts))
            value = supported / len(claims)
            return JudgeVerdict(value=value, samples=(value,), variance=0.0)
        sentences = split_claims(response)
        if len(sentences) < 2:
            return JudgeVerdict(value=1.0, samples=(1.0,), variance=0.0)
        distinct = len({normalize_answer(sentence) for sentence in sentences})
        value = distinct / len(sentences)
        return JudgeVerdict(value=value, samples=(value,), variance=0.0)
