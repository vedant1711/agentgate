"""Deterministic judge backends with programmable biases.

Two uses, both load-bearing:

* **Verification.** The bias-control machinery has to be tested against a judge whose bias is
  *known*, or the audits are only checked against themselves. A synthetic judge with a
  programmed 30% slot-A preference must produce a measured flip rate of 30%.
* **The Judge Bias Lab (K2).** The demo's "mitigations OFF" toggle needs to show what an
  unmitigated judge does on real data, offline and for free.

These are honest fixtures, not stand-ins for a real judge: they compute a deterministic notion
of quality and then distort it in exactly the ways the literature documents.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass

from agentgate.judge.rubrics import LIKERT_MAX, LIKERT_MIN, denormalise
from agentgate.metrics.lexical_judge import content_words, support_score
from agentgate.providers.mock import usage_for
from agentgate.providers.types import ChatRequest, ChatResponse
from agentgate.seeds import derive_seed

_RESPONSE_BLOCK = re.compile(
    r"### Assistant response\n(.*?)(?=\n### |\nRespond with JSON|\Z)", re.S
)
_REFERENCE_BLOCK = re.compile(r"### Reference answer\n(.*?)(?=\n### |\nRespond with JSON|\Z)", re.S)
_CONTEXT_BLOCK = re.compile(r"### Retrieved context\n(.*?)(?=\n### |\nRespond with JSON|\Z)", re.S)
_TASK_BLOCK = re.compile(r"### Task\n(.*?)(?=\n### |\Z)", re.S)
_SLOT_A = re.compile(r"### Response A\n(.*?)(?=\n### |\Z)", re.S)
_SLOT_B = re.compile(r"### Response B\n(.*?)(?=\n### |\Z)", re.S)
_CRITERION = re.compile(r"evaluating one criterion: \*\*(\w+)\*\*")

VERBOSITY_SATURATION_CHARS = 500.0
"""Length at which the verbosity bonus saturates — beyond this, padding stops helping."""


def _block(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def true_quality(response: str, reference: str, contexts: list[str]) -> float:
    """A deterministic notion of quality in [0, 1], independent of length and slot.

    Deliberately crude — its job is to be *stable and unbiased*, so any measured bias in the
    synthetic judge is the bias that was programmed in and nothing else.
    """
    if reference:
        return support_score(reference, [response])
    if contexts:
        return support_score(response, contexts)
    words = content_words(response)
    return min(1.0, len(words) / 12.0)


@dataclass(slots=True)
class SyntheticJudge:
    """A judge "model" whose biases are known exactly.

    Args:
        position_bias: Probability of answering slot ``A`` regardless of content. Applied
            identically in both slot orders, which makes the resulting **flip rate equal to this
            value**: an anchored preference for slot A disagrees with itself exactly when the
            slots are swapped.
        verbosity_bias: Score bonus, in normalised units, for a maximally long response.
        markdown_bias: Score bonus for markdown-dense responses.
        noise: Per-draw jitter, so J samples differ and the judge reports non-zero measurement
            variance the way a real one does.
        seed: Root seed for every deterministic decision.
    """

    position_bias: float = 0.0
    verbosity_bias: float = 0.0
    markdown_bias: float = 0.0
    noise: float = 0.05
    seed: int = 20260101
    name: str = "synthetic"

    def __call__(self, request: ChatRequest) -> ChatResponse:
        """Answer a rubric or pairwise judging request."""
        user = next((m.content for m in request.messages if m.role == "user"), "")
        payload = (
            self._pairwise(user) if "### Response A" in user else self._rubric(user, request.seed)
        )
        text = json.dumps(payload)
        return ChatResponse(
            text=text,
            model=request.model,
            provider="synthetic-judge",
            latency_ms=6.0,
            usage=usage_for(request, text),
        )

    # -- rubric ------------------------------------------------------------

    def _rubric(self, user: str, seed: int | None) -> dict[str, object]:
        response = _block(_RESPONSE_BLOCK, user)
        reference = _block(_REFERENCE_BLOCK, user)
        context = _block(_CONTEXT_BLOCK, user)
        contexts = [context] if context and context != "(no context retrieved)" else []

        quality = true_quality(response, reference, contexts)
        quality += self.verbosity_bias * min(1.0, len(response) / VERBOSITY_SATURATION_CHARS)
        quality += self.markdown_bias * min(1.0, _markdown_hits(response) / 10.0)
        if self.noise:
            rng = random.Random(derive_seed(self.seed, "rubric", response[:64], seed or 0))
            quality += rng.uniform(-self.noise, self.noise)

        raw = round(denormalise(min(1.0, max(0.0, quality))))
        criterion = _CRITERION.search(user)
        return {
            "reasoning": (
                f"Assessed {criterion.group(1) if criterion else 'the criterion'} from content "
                f"overlap; length and formatting were {'' if self.biased else 'not '}considered."
            ),
            "score": int(min(LIKERT_MAX, max(LIKERT_MIN, raw))),
            "evidence": [response[:80]] if response else [],
        }

    # -- pairwise ----------------------------------------------------------

    def _pairwise(self, user: str) -> dict[str, object]:
        task = _block(_TASK_BLOCK, user)
        slot_a = _block(_SLOT_A, user)
        slot_b = _block(_SLOT_B, user)

        # The bias decision is keyed on the *unordered* pair, so the same item is judged with the
        # same slot anchoring in both orders. That is what makes flip rate == position_bias.
        pair_key = "".join(sorted([slot_a[:120], slot_b[:120]]))
        rng = random.Random(derive_seed(self.seed, "pairwise", task[:80], pair_key))
        if rng.random() < self.position_bias:
            return {"reasoning": "Slot A read better.", "winner": "A"}

        quality_a = true_quality(slot_a, "", [])
        quality_b = true_quality(slot_b, "", [])
        if math.isclose(quality_a, quality_b, abs_tol=1e-9):
            winner = "tie"
        else:
            winner = "A" if quality_a > quality_b else "B"
        return {"reasoning": "Compared content coverage only.", "winner": winner}

    @property
    def biased(self) -> bool:
        """True when any bias knob is non-zero."""
        return bool(self.position_bias or self.verbosity_bias or self.markdown_bias)


def _markdown_hits(text: str) -> int:
    from agentgate.judge.audits import markdown_density

    return int(markdown_density(text) * len(text) / 100.0)


@dataclass(slots=True)
class MalformedJudge:
    """A judge that emits unparseable output, to prove a parse failure is flagged not zeroed."""

    fail_first: int = 99
    name: str = "malformed"
    _calls: int = 0

    def __call__(self, request: ChatRequest) -> ChatResponse:
        """Return garbage for the first ``fail_first`` calls, then valid JSON."""
        self._calls += 1
        text = (
            "I think this deserves maybe a four out of five?"
            if self._calls <= self.fail_first
            else json.dumps({"reasoning": "ok", "score": 4})
        )
        return ChatResponse(
            text=text,
            model=request.model,
            provider="malformed-judge",
            latency_ms=4.0,
            usage=usage_for(request, text),
        )
