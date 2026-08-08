"""Rubric definitions for the absolute judges (D1).

G-Eval's finding (arXiv:2303.16634) is the design constraint: a decomposed criterion with
anchored level descriptions and elicited reasoning aligns with humans far better than "rate the
quality 1-10". So each criterion here is a *separate* judge with its own anchored 1-5 scale, its
own evidence requirement, and its own structured output — and AgentGate never computes a blended
quality number anywhere.

Rubrics are content-hashed into ``agentgate.lock``. Editing a rubric changes what a score means,
so it must invalidate comparison against history exactly the way a model change does.
"""

from __future__ import annotations

from typing import Any, Final

from pydantic import Field

from agentgate.schemas.common import FrozenModel

LIKERT_MIN: Final = 1
LIKERT_MAX: Final = 5


def normalise(score: float) -> float:
    """Map an anchored 1-5 Likert score onto [0, 1].

    Args:
        score: Raw judge score.

    Returns:
        ``(score - 1) / 4``, clamped into [0, 1] so a judge that ignores the scale cannot push a
        proportion metric out of range.
    """
    return min(1.0, max(0.0, (score - LIKERT_MIN) / (LIKERT_MAX - LIKERT_MIN)))


def denormalise(value: float) -> float:
    """Inverse of :func:`normalise`, for reporting scores on the original scale."""
    return value * (LIKERT_MAX - LIKERT_MIN) + LIKERT_MIN


class Rubric(FrozenModel):
    """One criterion, its question, and its anchored level descriptions."""

    criterion: str
    question: str
    anchors: dict[int, str] = Field(description="Level description per Likert point 1-5.")
    needs_reference: bool = False
    needs_contexts: bool = False
    note: str = ""

    def render(
        self,
        *,
        prompt: str,
        response: str,
        reference: str = "",
        contexts: list[str] | None = None,
    ) -> str:
        """Build the judge's user message.

        Chain-of-thought is elicited *before* the score, because a score written first becomes a
        conclusion the reasoning is then made to justify.
        """
        blocks = [
            f"You are evaluating one criterion: **{self.criterion}**.",
            f"Question: {self.question}",
            "Scale:",
            *(f"  {level} = {self.anchors[level]}" for level in sorted(self.anchors)),
            "",
            f"### Task given to the assistant\n{prompt}",
            f"### Assistant response\n{response}",
        ]
        if self.needs_reference:
            blocks.append(f"### Reference answer\n{reference or '(none provided)'}")
        if self.needs_contexts:
            joined = "\n\n".join(contexts or []) or "(no context retrieved)"
            blocks.append(f"### Retrieved context\n{joined}")
        blocks.append(
            "Respond with JSON only: "
            '{"reasoning": "<brief reasoning, written before you decide>", '
            '"score": <integer 1-5>, "evidence": ["<quote from the response>", ...]}'
        )
        return "\n\n".join(blocks)

    def system_prompt(self) -> str:
        """Instruction framing shared by every rubric judge."""
        return (
            "You are a strict, impartial evaluator. Judge only the criterion you are given; "
            "ignore length, formatting, and confidence of tone. Longer is not better. "
            "Reason briefly first, then give an integer score. Output JSON only."
        )


OUTPUT_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "required": ["reasoning", "score"],
    "properties": {
        "reasoning": {"type": "string"},
        "score": {"type": "integer", "minimum": LIKERT_MIN, "maximum": LIKERT_MAX},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}
"""JSON Schema constraining rubric-judge output, passed to providers that support it."""


RUBRICS: Final[dict[str, Rubric]] = {
    "correctness": Rubric(
        criterion="correctness",
        question="Are the response's factual claims correct relative to the reference answer?",
        needs_reference=True,
        anchors={
            1: "Contradicts the reference on the central claim.",
            2: "Mostly wrong; one incidental detail happens to match.",
            3: "Partly right: the central claim is right but a supporting detail is wrong.",
            4: "Right, with a minor imprecision that does not mislead.",
            5: "Fully consistent with the reference on every claim it makes.",
        },
    ),
    "completeness": Rubric(
        criterion="completeness",
        question="Does the response cover everything the reference answer covers?",
        needs_reference=True,
        anchors={
            1: "Addresses none of the reference's content.",
            2: "Addresses a small fraction and omits the main point.",
            3: "Covers the main point but omits required supporting elements.",
            4: "Covers nearly everything; one non-essential element is missing.",
            5: "Covers every element the reference covers.",
        },
    ),
    "instruction_following": Rubric(
        criterion="instruction_following",
        question="Did the response do what the instruction actually asked, in the form asked?",
        anchors={
            1: "Ignores the instruction or answers a different question.",
            2: "Addresses the topic but violates explicit constraints (format, scope, limits).",
            3: "Follows the main instruction but misses a stated constraint.",
            4: "Follows the instruction with a trivial deviation.",
            5: "Follows every stated instruction and constraint exactly.",
        },
    ),
    "coherence": Rubric(
        criterion="coherence",
        question="Is the response internally consistent, ordered, and free of repetition?",
        anchors={
            1: "Self-contradictory or incomprehensible.",
            2: "Disordered, with repeated or conflicting statements.",
            3: "Understandable but repetitive or loosely organised.",
            4: "Clear and consistent, with minor redundancy.",
            5: "Clear, ordered, non-repetitive, and internally consistent.",
        },
        note="A verbosity attack should lower this even as it inflates naive preference (E4).",
    ),
    "faithfulness": Rubric(
        criterion="faithfulness",
        question="Is every claim in the response supported by the retrieved context?",
        needs_contexts=True,
        anchors={
            1: "The central claim is unsupported by, or contradicts, the context.",
            2: "Most claims are unsupported.",
            3: "About half the claims are supported; the rest are not.",
            4: "Nearly all claims are supported; one is an unsupported inference.",
            5: "Every claim is directly supported by the retrieved context.",
        },
    ),
}
"""The five absolute criteria. Adding one here is how a suite gains a new judge-backed metric."""


PAIRWISE_SYSTEM: Final = (
    "You are comparing two assistant responses to the same task. Judge which better satisfies "
    "the task. Ignore length, formatting, and tone; a longer answer is not a better answer. "
    "Respond with JSON only: "
    '{"reasoning": "<brief>", "winner": "A" | "B" | "tie"}'
)

PAIRWISE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "required": ["winner"],
    "properties": {
        "reasoning": {"type": "string"},
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
    },
    "additionalProperties": False,
}


def render_pairwise(prompt: str, first: str, second: str) -> str:
    """Build the pairwise judge's user message for a given slot order."""
    return (
        f"### Task\n{prompt}\n\n"
        f"### Response A\n{first}\n\n"
        f"### Response B\n{second}\n\n"
        "### Instruction\nWhich response better satisfies the task? Output JSON only."
    )


def rubrics_hash() -> str:
    """Content hash over every rubric, recorded in ``agentgate.lock``."""
    from agentgate.schemas.common import stable_hash

    return stable_hash({name: rubric.model_dump(mode="json") for name, rubric in RUBRICS.items()})
