"""Judge transcripts: every judge measurement, with its spread, persisted.

Judging is asynchronous and costly; metric scoring is synchronous and free. So the judge runs as
its own pass and writes a **transcript**, which the metrics layer then reads through a plain
synchronous :class:`~agentgate.metrics.base.Judge`.

That split buys three things beyond tidiness:

* **Uncertainty is kept, not collapsed.** Each entry stores all J samples, so C5 can fold the
  judge's within-item variance into the metric's standard error instead of pretending a mean is
  a measurement.
* **Auditability.** Every score keeps its elicited reasoning and quoted evidence.
* **Replayability.** A transcript is data; the demo ships one and re-analyses it offline.
"""

from __future__ import annotations

import statistics
from datetime import datetime
from typing import Any

from pydantic import Field

from agentgate.metrics.base import JudgeVerdict
from agentgate.schemas.common import AgentGateModel, FrozenModel, stable_hash


def item_key(
    criterion: str,
    prompt: str,
    response: str,
    *,
    reference: str = "",
    contexts: list[str] | None = None,
) -> str:
    """Stable identity for one judged item.

    Keyed on everything the judge sees, so changing the reference or the retrieved context
    produces a different item rather than silently reusing a stale verdict.
    """
    return stable_hash([criterion, prompt, response, reference, contexts or []])


class JudgeSample(FrozenModel):
    """One of the J draws for an item (D1: J=3 at temperature > 0)."""

    raw_score: float = Field(description="Judge's anchored 1-5 score.")
    score: float = Field(ge=0.0, le=1.0, description="Normalised to [0, 1].")
    reasoning: str = ""
    evidence: list[str] = Field(default_factory=list)
    tokens: int = 0


class JudgeEntry(FrozenModel):
    """All draws for one (criterion, item), plus what went wrong getting them."""

    key: str
    criterion: str
    samples: list[JudgeSample] = Field(default_factory=list)
    parse_failures: int = Field(
        default=0,
        description="Malformed judge replies retried. A sample that never parsed is flagged, "
        "never silently scored 0 (H1).",
    )
    flagged: bool = Field(
        default=False, description="True when no draw parsed; the metric must skip this item."
    )

    @property
    def mean(self) -> float:
        """Mean normalised score across draws."""
        return statistics.fmean(sample.score for sample in self.samples) if self.samples else 0.0

    @property
    def variance(self) -> float:
        """Within-item variance across draws — the quantity C5 propagates."""
        if len(self.samples) < 2:
            return 0.0
        return statistics.variance([sample.score for sample in self.samples])

    @property
    def tokens(self) -> int:
        """Tokens spent judging this item."""
        return sum(sample.tokens for sample in self.samples)

    def to_verdict(self) -> JudgeVerdict:
        """Render as the value object the metrics layer consumes."""
        return JudgeVerdict(
            value=self.mean,
            samples=tuple(sample.score for sample in self.samples),
            variance=self.variance,
            cost_tokens=self.tokens,
            rationale=self.samples[0].reasoning if self.samples else "",
        )


class PairwiseEntry(FrozenModel):
    """One A/B comparison, judged in both slot orders (D2)."""

    key: str
    forward_winner: str = Field(description="Winner with candidate in slot A: A | B | tie.")
    swapped_winner: str = Field(description="Winner with candidate in slot B: A | B | tie.")
    reasoning: str = ""

    @property
    def flipped(self) -> bool:
        """True when swapping slots changed the verdict — pure position effect."""
        mirrored = {"A": "B", "B": "A", "tie": "tie"}[self.swapped_winner]
        return mirrored != self.forward_winner

    @property
    def averaged_score(self) -> float:
        """Candidate's win share averaged over both orders: 1.0 win, 0.5 tie, 0.0 loss.

        The candidate occupies slot A forward and slot B swapped, so a consistent judge gives
        ``A`` then ``B``.
        """
        forward = {"A": 1.0, "tie": 0.5, "B": 0.0}[self.forward_winner]
        swapped = {"B": 1.0, "tie": 0.5, "A": 0.0}[self.swapped_winner]
        return (forward + swapped) / 2.0


class JudgeTranscript(AgentGateModel):
    """Every judge measurement made for one run."""

    judge_name: str = "judge"
    judge_model: str = ""
    temperature: float = 0.3
    n_samples: int = Field(default=3, ge=1)
    created_at: datetime | None = None
    entries: dict[str, JudgeEntry] = Field(default_factory=dict)
    pairwise: dict[str, PairwiseEntry] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def record(self, entry: JudgeEntry) -> None:
        """Store an entry under its key."""
        self.entries[entry.key] = entry

    def lookup(
        self,
        criterion: str,
        prompt: str,
        response: str,
        *,
        reference: str = "",
        contexts: list[str] | None = None,
    ) -> JudgeEntry | None:
        """Find the entry for one judged item, or ``None``."""
        return self.entries.get(
            item_key(criterion, prompt, response, reference=reference, contexts=contexts)
        )

    @property
    def total_tokens(self) -> int:
        """Tokens the whole judging pass spent."""
        return sum(entry.tokens for entry in self.entries.values())

    @property
    def flagged_items(self) -> list[str]:
        """Items whose judge output never parsed."""
        return sorted(key for key, entry in self.entries.items() if entry.flagged)

    @property
    def position_flip_rate(self) -> float | None:
        """Share of pairwise comparisons whose verdict flipped on slot swap (D2)."""
        if not self.pairwise:
            return None
        flips = sum(1 for entry in self.pairwise.values() if entry.flipped)
        return flips / len(self.pairwise)

    def criteria(self) -> list[str]:
        """Criteria present in this transcript, sorted."""
        return sorted({entry.criterion for entry in self.entries.values()})

    def scores_for(self, criterion: str) -> dict[str, float]:
        """``{item key: mean score}`` for one criterion."""
        return {
            key: entry.mean
            for key, entry in self.entries.items()
            if entry.criterion == criterion and not entry.flagged
        }

    def merge(self, other: JudgeTranscript) -> None:
        """Fold another transcript's entries into this one."""
        self.entries.update(other.entries)
        self.pairwise.update(other.pairwise)
        self.warnings.extend(other.warnings)


class JudgedItem(FrozenModel):
    """One thing to judge: what the agent was asked, and what it answered."""

    prompt: str
    response: str
    reference: str = ""
    contexts: list[str] = Field(default_factory=list)
    task_id: str = ""
    rep: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def key_for(self, criterion: str) -> str:
        """This item's transcript key for one criterion."""
        return item_key(
            criterion,
            self.prompt,
            self.response,
            reference=self.reference,
            contexts=list(self.contexts),
        )
