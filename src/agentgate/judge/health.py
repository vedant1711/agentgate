"""Assemble the judge-health panel that accompanies every judge-backed number (E4).

One object answers the question a reader should always ask: *how much should I discount this?*
Agreement with humans, position stability, verbosity and format sensitivity, drift status, and
whether the judge is independent of the agent — all in one place, and all published rather than
kept in a config file.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentgate.judge.audits import JudgeAudit, audit_scores
from agentgate.judge.calibration import CalibrationReport
from agentgate.judge.drift import DriftReport
from agentgate.judge.transcript import JudgeTranscript
from agentgate.schemas.results import JudgeHealth


@dataclass(slots=True)
class HealthInputs:
    """Everything needed to build a health panel."""

    transcript: JudgeTranscript
    responses: dict[str, str]
    """``{item key: response text}`` for the audits."""
    calibration: CalibrationReport | None = None
    drift: DriftReport | None = None
    self_judging: bool = False


def audit_transcript(
    transcript: JudgeTranscript, responses: dict[str, str], *, criterion: str | None = None
) -> JudgeAudit:
    """Run the bias audits over one criterion (or the most-covered one).

    Args:
        transcript: Judge measurements.
        responses: Response text per item key.
        criterion: Criterion to audit; defaults to whichever has the most scored items.

    Returns:
        The audit panel.
    """
    criteria = transcript.criteria()
    chosen = criterion or (
        max(criteria, key=lambda name: len(transcript.scores_for(name))) if criteria else ""
    )
    return audit_scores(
        transcript.scores_for(chosen) if chosen else {},
        responses,
        position_flip_rate=transcript.position_flip_rate,
    )


def build_health(inputs: HealthInputs, *, criterion: str | None = None) -> JudgeHealth:
    """Build the :class:`~agentgate.schemas.results.JudgeHealth` panel for a comparison.

    Args:
        inputs: Transcript, responses, and optional calibration/drift reports.
        criterion: Criterion to audit for bias; defaults to the best-covered one.

    Returns:
        The panel, carrying every warning any subsystem raised.
    """
    audit = audit_transcript(inputs.transcript, inputs.responses, criterion=criterion)
    calibration = inputs.calibration
    drift = inputs.drift

    warnings: list[str] = list(inputs.transcript.warnings)
    warnings.extend(audit.warnings)
    if calibration is not None:
        warnings.extend(calibration.warnings)
    if drift is not None:
        warnings.extend(drift.warnings)
    if calibration is None:
        warnings.append(
            "no calibration set: this judge has never been checked against a human, so its "
            "agreement is unknown and it must not back a gated metric (D4)"
        )

    best = None
    if calibration is not None:
        best = calibration.criterion(criterion) if criterion else None
        if best is None and calibration.per_criterion:
            best = max(
                calibration.per_criterion,
                key=lambda entry: entry.cohens_kappa if entry.cohens_kappa is not None else -2.0,
            )

    return JudgeHealth(
        judge_model=inputs.transcript.judge_model,
        cohens_kappa=best.cohens_kappa if best else None,
        spearman_rho=best.spearman_rho if best else None,
        n_calibration_items=calibration.n_labels if calibration else 0,
        position_flip_rate=inputs.transcript.position_flip_rate,
        verbosity_correlation=audit.verbosity.rho,
        markdown_correlation=audit.markdown.rho,
        drift_detected=bool(drift and drift.detected),
        drift_detail=drift.banner if drift else "",
        self_judging=inputs.self_judging,
        warnings=warnings,
    )


def responses_from_items(items: list[tuple[str, str]]) -> dict[str, str]:
    """Build the ``{key: response}`` map the audits need from ``(key, response)`` pairs."""
    return dict(items)
