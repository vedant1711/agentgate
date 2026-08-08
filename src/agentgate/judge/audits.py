"""Judge bias audits (D3).

You cannot remove judge bias, but you can *measure* it and publish the number. Three audits run
on every judge-backed suite:

* **Verbosity** — Spearman correlation between response length and judge score. Wang et al. and
  MT-Bench measure 15-30 point preference inflation for longer answers; a correlation above 0.4
  here means this judge is partly scoring length on this task type.
* **Format** — the same procedure on markdown density, because a judge that rewards bullet
  points is rewarding a formatter, not an agent.
* **Position** — the pairwise flip rate (computed in the transcript), flagged above 20%.

When verbosity correlation trips, a **length-controlled re-analysis** is offered: regress score
on length and gate on the residuals. It is reported alongside the raw scores, never silently
substituted — a residual is a different quantity and must be labelled as one.
"""

from __future__ import annotations

import math
import re
from typing import Final

from pydantic import Field

from agentgate.schemas.common import FrozenModel

VERBOSITY_THRESHOLD: Final = 0.4
"""|rho| above this flags the judge as length-sensitive on this suite (D3)."""

POSITION_FLIP_THRESHOLD: Final = 0.20
"""Flip rate above this flags the judge as position-unstable for this task type (D2)."""

_MARKDOWN_MARKERS = re.compile(r"(^|\n)\s*([-*+]\s|\d+\.\s|#{1,6}\s|>\s)|\*\*|`{1,3}|\|")


def markdown_density(text: str) -> float:
    """Markdown markers per 100 characters.

    Args:
        text: Response text.

    Returns:
        Marker density; 0.0 for empty text.
    """
    if not text:
        return 0.0
    return 100.0 * len(_MARKDOWN_MARKERS.findall(text)) / len(text)


def rank(values: list[float]) -> list[float]:
    """Rank values ascending, averaging ties (the convention Spearman assumes)."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        shared = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = shared
        position = end + 1
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation, returning 0.0 when either series is constant."""
    n = len(xs)
    if n < 2 or len(ys) != n:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [value - mean_x for value in xs]
    dy = [value - mean_y for value in ys]
    denominator = math.sqrt(sum(v * v for v in dx)) * math.sqrt(sum(v * v for v in dy))
    if denominator == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(dx, dy, strict=True)) / denominator


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation — Pearson on ranks, tie-corrected by average ranking."""
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    return pearson(rank(xs), rank(ys))


class Correlation(FrozenModel):
    """One audit correlation with its sample size and verdict."""

    feature: str
    rho: float
    n: int = Field(ge=0)
    threshold: float = VERBOSITY_THRESHOLD

    @property
    def flagged(self) -> bool:
        """True when the correlation exceeds its threshold in magnitude."""
        return self.n >= 3 and abs(self.rho) > self.threshold

    def describe(self) -> str:
        """One-line summary for the judge-health panel."""
        verdict = "FLAGGED" if self.flagged else "ok"
        return f"{self.feature}: rho={self.rho:+.3f} (n={self.n}, |rho|>{self.threshold}) {verdict}"


class LengthControlled(FrozenModel):
    """Judge scores with the length effect regressed out (D3)."""

    slope: float
    intercept: float
    residuals: dict[str, float] = Field(default_factory=dict)
    note: str = (
        "Residual scores after regressing judge score on response length. Reported beside the "
        "raw scores, never instead of them — a residual is a different quantity."
    )


class JudgeAudit(FrozenModel):
    """The full bias panel published next to every judge-backed number."""

    n_items: int = Field(ge=0)
    verbosity: Correlation
    markdown: Correlation
    position_flip_rate: float | None = None
    length_controlled: LengthControlled | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def clean(self) -> bool:
        """True when no audit tripped."""
        return not self.warnings


def audit_scores(
    scores: dict[str, float],
    responses: dict[str, str],
    *,
    position_flip_rate: float | None = None,
    control_for_length: bool = True,
) -> JudgeAudit:
    """Run the verbosity, format, and position audits over one criterion's scores.

    Args:
        scores: ``{item key: judge score}``.
        responses: ``{item key: response text}``.
        position_flip_rate: From the pairwise transcript, when one exists.
        control_for_length: Compute the length-controlled re-analysis when verbosity trips.

    Returns:
        The audit panel, with a warning per tripped check.
    """
    keys = [key for key in sorted(scores) if key in responses]
    values = [scores[key] for key in keys]
    lengths = [float(len(responses[key])) for key in keys]
    densities = [markdown_density(responses[key]) for key in keys]

    verbosity = Correlation(feature="response_length", rho=spearman(lengths, values), n=len(keys))
    markdown = Correlation(feature="markdown_density", rho=spearman(densities, values), n=len(keys))

    warnings: list[str] = []
    if verbosity.flagged:
        warnings.append(
            f"VERBOSITY BIAS: judge score correlates with response length "
            f"(Spearman rho={verbosity.rho:+.2f}, n={verbosity.n}). Literature reports 15-30 "
            f"points of preference inflation for longer answers; treat this judge's absolute "
            f"scores with caution and prefer the length-controlled residuals."
        )
    if markdown.flagged:
        warnings.append(
            f"FORMAT BIAS: judge score correlates with markdown density "
            f"(Spearman rho={markdown.rho:+.2f}, n={markdown.n}). The judge is partly rewarding "
            f"formatting rather than substance."
        )
    if position_flip_rate is not None and position_flip_rate > POSITION_FLIP_THRESHOLD:
        warnings.append(
            f"POSITION INSTABILITY: {position_flip_rate:.0%} of pairwise verdicts flip when the "
            f"slots are swapped (threshold {POSITION_FLIP_THRESHOLD:.0%}). Pairwise summaries "
            f"for this task type are not trustworthy."
        )

    controlled = None
    if control_for_length and verbosity.flagged and len(keys) >= 3:
        controlled = _regress_out(keys, lengths, values)

    return JudgeAudit(
        n_items=len(keys),
        verbosity=verbosity,
        markdown=markdown,
        position_flip_rate=position_flip_rate,
        length_controlled=controlled,
        warnings=warnings,
    )


def _regress_out(keys: list[str], lengths: list[float], values: list[float]) -> LengthControlled:
    """Ordinary least squares of score on length; residuals are the length-free scores."""
    n = len(lengths)
    mean_x = sum(lengths) / n
    mean_y = sum(values) / n
    variance = sum((x - mean_x) ** 2 for x in lengths)
    slope = (
        sum((x - mean_x) * (y - mean_y) for x, y in zip(lengths, values, strict=True)) / variance
        if variance
        else 0.0
    )
    intercept = mean_y - slope * mean_x
    residuals = {
        key: value - (intercept + slope * length)
        for key, length, value in zip(keys, lengths, values, strict=True)
    }
    return LengthControlled(slope=slope, intercept=intercept, residuals=residuals)
