"""A synchronous :class:`~agentgate.metrics.base.Judge` backed by a recorded transcript.

The metrics engine is synchronous and must stay that way — scoring should be a pure function of
recorded data. This adapter is what lets an asynchronous, network-bound judging pass sit behind
that interface: the transcript is computed once, then read many times, offline and for free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentgate.errors import JudgeError
from agentgate.judge.transcript import JudgeTranscript
from agentgate.metrics.base import Judge, JudgeVerdict
from agentgate.metrics.lexical_judge import LexicalJudge


@dataclass
class TranscriptJudge:
    """Serves rubric verdicts from a transcript, delegating claim work to a fallback.

    Args:
        transcript: Recorded judge measurements.
        fallback: Handles claim extraction and support checks, which the rubric pass does not
            cover. Defaults to the deterministic :class:`LexicalJudge`; the report always names
            which judge produced which number.
        strict: When True, an item missing from the transcript raises instead of falling back —
            used by the gate, where a silently substituted judge would be a false comparison.
    """

    transcript: JudgeTranscript
    fallback: Judge | None = field(default_factory=LexicalJudge)
    strict: bool = False

    @property
    def name(self) -> str:
        """Judge identity recorded next to every judge-backed number."""
        return f"{self.transcript.judge_name}:{self.transcript.judge_model}"

    def score_criterion(
        self,
        criterion: str,
        prompt: str,
        response: str,
        *,
        reference: str = "",
        contexts: list[str] | None = None,
    ) -> JudgeVerdict:
        """Return the recorded verdict for this item.

        Raises:
            JudgeError: When the item is absent and no fallback is permitted.
        """
        entry = self.transcript.lookup(
            criterion, prompt, response, reference=reference, contexts=contexts
        )
        if entry is not None and not entry.flagged:
            return entry.to_verdict()
        if entry is not None and entry.flagged:
            msg = (
                f"judge output for criterion {criterion!r} never parsed after "
                f"{entry.parse_failures} attempt(s); this item is flagged, not scored"
            )
            raise JudgeError(msg)
        if self.strict or self.fallback is None:
            msg = f"no recorded judge verdict for criterion {criterion!r} on this item"
            raise JudgeError(msg)
        return self.fallback.score_criterion(
            criterion, prompt, response, reference=reference, contexts=contexts
        )

    def extract_claims(self, text: str) -> list[str]:
        """Decompose text into atomic claims via the fallback judge."""
        if self.fallback is None:
            msg = "claim extraction needs a fallback judge"
            raise JudgeError(msg)
        return self.fallback.extract_claims(text)

    def check_claims(self, claims: list[str], evidence: list[str]) -> list[bool]:
        """Check claim support via the fallback judge."""
        if self.fallback is None:
            msg = "claim checking needs a fallback judge"
            raise JudgeError(msg)
        return self.fallback.check_claims(claims, evidence)
