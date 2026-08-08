"""Judge-human calibration (D4).

A judge that has never been checked against a human is an unvalidated instrument, and a gate
built on one is measuring the judge. So AgentGate requires a hand-labelled calibration set and
publishes two agreement statistics beside every judge-backed number:

* **Cohen's kappa** on the discretised 1-5 scale — chance-corrected categorical agreement.
* **Spearman rho** on the ordinal scores — does the judge rank items the way a human does?

``require_judge_kappa`` (default 0.6) is enforced by the gate: below it, a judge-backed metric
may be reported but may not gate. Kappa is computed by hand here rather than imported, because
it is gate-critical and its treatment of the marginals is the whole point of the statistic.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import Field

from agentgate.judge.audits import spearman
from agentgate.judge.rubrics import LIKERT_MAX, LIKERT_MIN, denormalise
from agentgate.judge.transcript import JudgeTranscript, item_key
from agentgate.schemas.common import AgentGateModel, FrozenModel

DEFAULT_LABEL_DIR = Path("datasets/calibration")
KAPPA_GATE_FLOOR = 0.6
"""Below this, a judge may inform a report but may not back a gated metric (D4)."""


class HumanLabel(FrozenModel):
    """One hand-assigned score, produced by ``agentgate label``."""

    criterion: str
    prompt: str
    response: str
    score: int = Field(ge=LIKERT_MIN, le=LIKERT_MAX)
    reference: str = ""
    contexts: list[str] = Field(default_factory=list)
    labeler: str = ""
    note: str = ""

    @property
    def key(self) -> str:
        """Transcript key this label corresponds to."""
        return item_key(
            self.criterion,
            self.prompt,
            self.response,
            reference=self.reference,
            contexts=list(self.contexts),
        )


class LabelSet(AgentGateModel):
    """A collection of human labels, persisted as JSONL."""

    name: str = "calibration"
    labels: list[HumanLabel] = Field(default_factory=list)

    def __len__(self) -> int:
        """Number of labels."""
        return len(self.labels)

    def criteria(self) -> list[str]:
        """Criteria covered by this set, sorted."""
        return sorted({label.criterion for label in self.labels})

    def for_criterion(self, criterion: str) -> list[HumanLabel]:
        """Labels for one criterion."""
        return [label for label in self.labels if label.criterion == criterion]

    @classmethod
    def load(cls, path: str | Path, *, name: str = "calibration") -> LabelSet:
        """Read a JSONL label file; a missing file yields an empty set."""
        source = Path(path)
        if not source.exists():
            return cls(name=name)
        labels = [
            HumanLabel.model_validate(json.loads(line))
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls(name=name, labels=labels)

    def save(self, path: str | Path) -> int:
        """Write the set as JSONL.

        Args:
            path: Destination file.

        Returns:
            Number of labels written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(label.model_dump_json() + "\n" for label in self.labels), encoding="utf-8"
        )
        return len(self.labels)

    def add(self, label: HumanLabel) -> None:
        """Append a label, replacing any existing one for the same item."""
        self.labels = [existing for existing in self.labels if existing.key != label.key]
        self.labels.append(label)


# ---------------------------------------------------------------------------
# Agreement statistics
# ---------------------------------------------------------------------------


def cohens_kappa(first: Sequence[int], second: Sequence[int]) -> float:
    """Cohen's kappa for two raters on the same items.

    ``kappa = (p_o - p_e) / (1 - p_e)`` where ``p_o`` is observed agreement and ``p_e`` is the
    agreement expected from the raters' *marginal* distributions. That correction is the point:
    two raters who both say "4" most of the time agree often by construction, and raw agreement
    would call that skill.

    Args:
        first: Rater A's categorical scores.
        second: Rater B's scores, aligned item-for-item.

    Returns:
        Kappa in [-1, 1]. Returns 1.0 for perfect agreement even when ``p_e == 1`` (both raters
        constant and identical), where the usual formula is 0/0.

    Raises:
        ValueError: When the two sequences differ in length or are empty.
    """
    if len(first) != len(second):
        msg = f"kappa needs aligned ratings: got {len(first)} and {len(second)}"
        raise ValueError(msg)
    n = len(first)
    if n == 0:
        msg = "kappa needs at least one rated item"
        raise ValueError(msg)

    observed = sum(1 for a, b in zip(first, second, strict=True) if a == b) / n
    counts_a = Counter(first)
    counts_b = Counter(second)
    expected = sum(
        (counts_a[category] / n) * (counts_b[category] / n)
        for category in set(counts_a) | set(counts_b)
    )
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def spearman_rho(first: Sequence[float], second: Sequence[float]) -> float:
    """Spearman rank correlation between two ordinal score series."""
    return spearman(list(first), list(second))


class CriterionAgreement(FrozenModel):
    """Judge-human agreement for one criterion."""

    criterion: str
    n: int = Field(ge=0)
    cohens_kappa: float | None = None
    spearman_rho: float | None = None
    mean_human: float | None = None
    mean_judge: float | None = None
    unmatched: int = Field(default=0, description="Labels with no corresponding transcript entry.")

    @property
    def meets_gate_bar(self) -> bool:
        """True when kappa clears the D4 floor for backing a gated metric."""
        return self.cohens_kappa is not None and self.cohens_kappa >= KAPPA_GATE_FLOOR

    @property
    def bias(self) -> float | None:
        """Judge mean minus human mean, on the 1-5 scale. Positive means the judge is generous."""
        if self.mean_judge is None or self.mean_human is None:
            return None
        return self.mean_judge - self.mean_human


class CalibrationReport(FrozenModel):
    """The full judge-human agreement panel (D4)."""

    judge_model: str = ""
    n_labels: int = Field(default=0, ge=0)
    per_criterion: list[CriterionAgreement] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def criterion(self, name: str) -> CriterionAgreement | None:
        """Agreement for one criterion, or ``None``."""
        return next((entry for entry in self.per_criterion if entry.criterion == name), None)

    @property
    def overall_kappa(self) -> float | None:
        """Sample-weighted mean kappa across criteria."""
        rated = [entry for entry in self.per_criterion if entry.cohens_kappa is not None]
        total = sum(entry.n for entry in rated)
        if not rated or total == 0:
            return None
        return sum(entry.cohens_kappa * entry.n for entry in rated if entry.cohens_kappa) / total

    def gated_criteria(self) -> list[str]:
        """Criteria whose agreement is high enough to back a gated metric."""
        return sorted(entry.criterion for entry in self.per_criterion if entry.meets_gate_bar)


def calibrate(labels: LabelSet, transcript: JudgeTranscript) -> CalibrationReport:
    """Compare a judge transcript against human labels.

    Judge means are compared on the human's own 1-5 scale (the normalised score is mapped back),
    and rounded to the nearest category for kappa — kappa is a categorical statistic and needs
    categories, not continuous means.

    Args:
        labels: Hand-assigned scores.
        transcript: The judge's measurements on the same items.

    Returns:
        Per-criterion agreement plus warnings for anything that fails the D4 bar.
    """
    report_criteria: list[CriterionAgreement] = []
    warnings: list[str] = []

    for criterion in labels.criteria():
        human_scores: list[int] = []
        judge_scores: list[float] = []
        unmatched = 0
        for label in labels.for_criterion(criterion):
            entry = transcript.entries.get(label.key)
            if entry is None or entry.flagged:
                unmatched += 1
                continue
            human_scores.append(label.score)
            judge_scores.append(denormalise(entry.mean))

        if not human_scores:
            report_criteria.append(
                CriterionAgreement(criterion=criterion, n=0, unmatched=unmatched)
            )
            warnings.append(f"{criterion}: no labelled item matched a judge verdict")
            continue

        rounded = [round(score) for score in judge_scores]
        agreement = CriterionAgreement(
            criterion=criterion,
            n=len(human_scores),
            cohens_kappa=cohens_kappa(human_scores, rounded),
            spearman_rho=spearman_rho([float(s) for s in human_scores], judge_scores),
            mean_human=sum(human_scores) / len(human_scores),
            mean_judge=sum(judge_scores) / len(judge_scores),
            unmatched=unmatched,
        )
        report_criteria.append(agreement)

        if not agreement.meets_gate_bar:
            warnings.append(
                f"{criterion}: Cohen's kappa {agreement.cohens_kappa:.2f} is below the "
                f"{KAPPA_GATE_FLOOR} floor. This criterion may be reported but must not back a "
                f"gated metric (D4)."
            )
        if agreement.n < 30:
            warnings.append(
                f"{criterion}: only {agreement.n} labelled items; kappa is itself imprecise at "
                f"this sample size. The spec calls for 60-100 calibration items."
            )

    return CalibrationReport(
        judge_model=transcript.judge_model,
        n_labels=len(labels),
        per_criterion=report_criteria,
        warnings=warnings,
    )


def items_needing_labels(
    transcript: JudgeTranscript, labels: LabelSet, *, limit: int = 50
) -> list[str]:
    """Transcript keys with no human label yet, for the labelling CLI to work through."""
    labelled = {label.key for label in labels.labels}
    return sorted(key for key in transcript.entries if key not in labelled)[:limit]


def load_label_sets(directory: str | Path = DEFAULT_LABEL_DIR) -> dict[str, LabelSet]:
    """Load every ``*.jsonl`` label set under ``directory``."""
    base = Path(directory)
    if not base.is_dir():
        return {}
    return {path.stem: LabelSet.load(path, name=path.stem) for path in sorted(base.glob("*.jsonl"))}


def merge_label_sets(sets: Iterable[LabelSet]) -> LabelSet:
    """Combine several label sets, later labels winning on conflict."""
    merged = LabelSet(name="merged")
    for label_set in sets:
        for label in label_set.labels:
            merged.add(label)
    return merged
