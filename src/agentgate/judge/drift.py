"""Judge-drift detection via a frozen anchor set (D4).

Providers update models behind stable ids. When that happens, every historical judge-backed
number silently stops being comparable — and nothing in a normal eval pipeline notices, because
the scores still look like scores.

So every run re-scores a **frozen anchor set** first: 30 fixed items with recorded expected
score distributions. If the anchor mean has moved outside its historical 95% band, the run
banner says so and trend charts annotate the discontinuity. The run still completes; the point
is that the reader knows the y-axis changed meaning.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import Field

from agentgate.judge.transcript import JudgeTranscript, item_key
from agentgate.schemas.common import AgentGateModel, FrozenModel, stable_hash

DEFAULT_ANCHOR_PATH = Path("datasets/anchors/anchors.jsonl")
DRIFT_Z = 1.96
"""Two-sided 95% band around the recorded anchor mean."""


class AnchorItem(FrozenModel):
    """One frozen anchor with the score distribution it produced when the band was recorded."""

    criterion: str
    prompt: str
    response: str
    reference: str = ""
    contexts: list[str] = Field(default_factory=list)
    expected_mean: float = Field(ge=0.0, le=1.0)
    expected_sd: float = Field(default=0.0, ge=0.0)
    note: str = ""

    @property
    def key(self) -> str:
        """Transcript key for this anchor."""
        return item_key(
            self.criterion,
            self.prompt,
            self.response,
            reference=self.reference,
            contexts=list(self.contexts),
        )


class AnchorSet(AgentGateModel):
    """The frozen anchor set, persisted as JSONL and hashed into ``agentgate.lock``."""

    items: list[AnchorItem] = Field(default_factory=list)

    def __len__(self) -> int:
        """Number of anchors."""
        return len(self.items)

    def content_hash(self) -> str:
        """Hash over every anchor; a changed anchor set invalidates the recorded band."""
        return stable_hash([item.model_dump(mode="json") for item in self.items])

    def criteria(self) -> list[str]:
        """Criteria the anchors cover."""
        return sorted({item.criterion for item in self.items})

    @classmethod
    def load(cls, path: str | Path = DEFAULT_ANCHOR_PATH) -> AnchorSet:
        """Read anchors from JSONL; a missing file yields an empty set."""
        source = Path(path)
        if not source.exists():
            return cls()
        return cls(
            items=[
                AnchorItem.model_validate(json.loads(line))
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        )

    def save(self, path: str | Path = DEFAULT_ANCHOR_PATH) -> int:
        """Write anchors as JSONL.

        Args:
            path: Destination file.

        Returns:
            Number of anchors written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(item.model_dump_json() + "\n" for item in self.items), encoding="utf-8"
        )
        return len(self.items)


class AnchorDelta(FrozenModel):
    """How far one criterion's anchor mean has moved."""

    criterion: str
    n: int = Field(ge=0)
    expected_mean: float
    observed_mean: float
    band_half_width: float = Field(ge=0.0)

    @property
    def delta(self) -> float:
        """Observed minus expected."""
        return self.observed_mean - self.expected_mean

    @property
    def drifted(self) -> bool:
        """True when the observed mean sits outside the recorded 95% band."""
        return abs(self.delta) > self.band_half_width

    def describe(self) -> str:
        """One-line summary for the run banner."""
        arrow = "up" if self.delta > 0 else "down"
        return (
            f"{self.criterion}: {self.observed_mean:.3f} vs {self.expected_mean:.3f} "
            f"({arrow} {abs(self.delta):.3f}, band +/-{self.band_half_width:.3f})"
        )


class DriftReport(FrozenModel):
    """Whether this run's judge still behaves like the one that produced history."""

    judge_model: str = ""
    anchor_hash: str = ""
    n_anchors: int = Field(default=0, ge=0)
    n_missing: int = Field(default=0, ge=0)
    deltas: list[AnchorDelta] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def detected(self) -> bool:
        """True when any criterion drifted outside its band."""
        return any(delta.drifted for delta in self.deltas)

    @property
    def banner(self) -> str:
        """The banner text a drifted run must display."""
        if not self.detected:
            return ""
        moved = ", ".join(delta.describe() for delta in self.deltas if delta.drifted)
        return (
            f"JUDGE DRIFT DETECTED - scores are not comparable to history. "
            f"Anchor means moved: {moved}"
        )


def check_drift(
    anchors: AnchorSet, transcript: JudgeTranscript, *, z: float = DRIFT_Z
) -> DriftReport:
    """Compare a run's anchor scores against their recorded band.

    The band is the standard error of the anchor *mean* under the recorded per-item spread, so
    it tightens as the anchor set grows — which is the right behaviour: more anchors should make
    drift easier to see, not harder.

    Args:
        anchors: The frozen anchor set.
        transcript: This run's judge measurements, which must include the anchors.
        z: Band width in standard errors.

    Returns:
        A report whose ``detected`` flag drives the run banner.
    """
    if not anchors.items:
        return DriftReport(
            judge_model=transcript.judge_model,
            warnings=["no anchor set configured; judge drift cannot be detected (D4)"],
        )

    by_criterion: dict[str, list[tuple[AnchorItem, float]]] = {}
    missing = 0
    for item in anchors.items:
        entry = transcript.entries.get(item.key)
        if entry is None or entry.flagged:
            missing += 1
            continue
        by_criterion.setdefault(item.criterion, []).append((item, entry.mean))

    deltas: list[AnchorDelta] = []
    for criterion, pairs in sorted(by_criterion.items()):
        observed = [score for _, score in pairs]
        expected = [item.expected_mean for item, _ in pairs]
        spreads = [item.expected_sd for item, _ in pairs]
        n = len(pairs)
        pooled_sd = math.sqrt(sum(sd * sd for sd in spreads) / n) if n else 0.0
        half_width = z * pooled_sd / math.sqrt(n) if n else 0.0
        deltas.append(
            AnchorDelta(
                criterion=criterion,
                n=n,
                expected_mean=sum(expected) / n,
                observed_mean=sum(observed) / n,
                band_half_width=half_width,
            )
        )

    warnings: list[str] = []
    if missing:
        warnings.append(
            f"{missing} anchor item(s) were not scored this run; drift detection is incomplete"
        )
    report = DriftReport(
        judge_model=transcript.judge_model,
        anchor_hash=anchors.content_hash(),
        n_anchors=len(anchors),
        n_missing=missing,
        deltas=deltas,
        warnings=warnings,
    )
    if report.detected:
        warnings.append(report.banner)
    return report.model_copy(update={"warnings": warnings})


def record_band(anchors: AnchorSet, transcript: JudgeTranscript) -> AnchorSet:
    """Re-record each anchor's expected mean and spread from a trusted transcript.

    Used once when establishing the baseline band, and again deliberately after an accepted
    judge change — never automatically, or drift detection would follow the drift.

    Args:
        anchors: Anchor definitions.
        transcript: The transcript whose scores become the new band.

    Returns:
        A new anchor set with updated expectations.
    """
    updated: list[AnchorItem] = []
    for item in anchors.items:
        entry = transcript.entries.get(item.key)
        if entry is None or entry.flagged:
            updated.append(item)
            continue
        updated.append(
            item.model_copy(
                update={
                    "expected_mean": entry.mean,
                    "expected_sd": math.sqrt(entry.variance) if entry.variance else 0.05,
                }
            )
        )
    return AnchorSet(items=updated)
