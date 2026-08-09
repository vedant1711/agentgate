"""The interactive demo page: the gate's decision, made tangible.

A reader can be told that a verdict depends on the non-inferiority margin and the sample size. It
lands differently when they drag a slider and watch REGRESSION become UNDERPOWERED become PASS on
real data, and see the interval and the p-values move with it.

**Every position on every slider is computed by the real engine, not by JavaScript.** The grid is
evaluated at build time with :func:`~agentgate.gate.engine.tests_at_margin`, the same function the
gate itself calls, and the page only looks values up. Reimplementing a paired t-test in browser
JavaScript would have been easier and would have quietly created a second, unverified
implementation of the one thing this project must get right — a demo that disagreed with the tool
it demonstrates would be worse than no demo.

The underlying data is real too: per-cluster paired differences produced by running the actual
pipeline over the actual suite, carried on
:attr:`~agentgate.schemas.results.MetricComparison.analysis_units`, which exists for precisely this.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentgate.gate.engine import GateResult, tests_at_margin
from agentgate.schemas.common import Verdict
from agentgate.schemas.policy import GatePolicy
from agentgate.schemas.results import MetricComparison

MARGIN_STEPS = 21
"""Slider positions per metric. Enough to feel continuous, small enough to embed."""

ALPHA_CHOICES = (0.01, 0.05, 0.10)


@dataclass(frozen=True, slots=True)
class GridPoint:
    """One evaluated (margin, alpha) position for one metric."""

    margin: float
    alpha: float
    verdict: str
    p_regression: float
    p_noninferiority: float
    mde: float | None
    power: float | None


@dataclass(slots=True)
class MetricPanel:
    """One metric's interactive panel: its real data plus the precomputed verdict grid."""

    metric: str
    direction: str
    n_units: int
    clustered: bool
    baseline: float
    candidate: float
    delta: float
    ci_low: float | None
    ci_high: float | None
    correlation: float | None
    units: list[float]
    default_margin: float
    grid: list[GridPoint] = field(default_factory=list)


def _verdict_at(
    comparison: MetricComparison, *, margin: float, alpha: float, policy: GatePolicy
) -> GridPoint:
    """Rule on one metric at one (margin, alpha), using the engine's own logic.

    The three-way rule, applied exactly as the gate applies it: a rejected one-sided test against
    the margin is a REGRESSION; non-inferiority *established* is a PASS; anything else is
    UNDERPOWERED, which is the honest answer rather than a default to green.
    """
    tuned = policy.model_copy(update={"alpha": alpha})
    regression, noninferiority, power = tests_at_margin(comparison, margin=margin, policy=tuned)

    if regression.p_one_sided <= alpha:
        verdict = Verdict.REGRESSION.value
    elif noninferiority.p_one_sided <= alpha:
        verdict = Verdict.PASS.value
    else:
        verdict = Verdict.UNDERPOWERED.value

    return GridPoint(
        margin=margin,
        alpha=alpha,
        verdict=verdict,
        p_regression=regression.p_one_sided,
        p_noninferiority=noninferiority.p_one_sided,
        mde=power.mde if power else None,
        power=power.achieved_power if power else None,
    )


def margin_steps(top: float, *, include: float | None = None) -> list[float]:
    """Evenly spaced slider positions from 0 to ``top``, guaranteed to contain ``include``.

    The policy's own margin must be a reachable slider position. Without it the demo can show
    every verdict except the one the gate actually reached, which is the single most useful thing
    a reader can compare against — and it would leave the page unable to prove it agrees with the
    tool it demonstrates.

    The nearest linear step is *replaced* rather than appended, so the grid stays exactly
    ``MARGIN_STEPS`` positions wide and the slider keeps a fixed range. The two endpoints are
    never replaced: zero must stay reachable because "demand exact equality and everything becomes
    UNDERPOWERED" is the demo's central lesson, and the top must stay put so the slider's range
    does not silently shrink.
    """
    steps = [index * top / (MARGIN_STEPS - 1) for index in range(MARGIN_STEPS)]
    if include is None or not 0.0 <= include <= top or include in steps:
        return steps
    interior = range(1, MARGIN_STEPS - 1)
    nearest = min(interior, key=lambda index: abs(steps[index] - include))
    steps[nearest] = include
    return sorted(steps)


def build_panel(
    comparison: MetricComparison, *, policy: GatePolicy, max_margin: float | None = None
) -> MetricPanel:
    """Evaluate one metric across the whole slider grid.

    Args:
        comparison: A real paired comparison carrying its analysis units.
        policy: The gate policy supplying normality and permutation settings.
        max_margin: Largest margin to evaluate. Defaults to a range wide enough that the verdict
            actually changes — a slider whose whole travel gives one answer teaches nothing.

    Returns:
        The panel, with ``MARGIN_STEPS x len(ALPHA_CHOICES)`` evaluated points.
    """
    units = list(comparison.analysis_units)
    spread = max(abs(comparison.delta.value), 0.01)
    top = max_margin if max_margin is not None else max(spread * 3.0, 0.05)

    margins = margin_steps(top, include=comparison.margin if max_margin is None else None)
    grid = [
        _verdict_at(comparison, margin=margin, alpha=alpha, policy=policy)
        for margin in margins
        for alpha in ALPHA_CHOICES
    ]
    return MetricPanel(
        metric=comparison.metric,
        direction=comparison.direction,
        n_units=comparison.n_pairs,
        clustered=comparison.clustered,
        baseline=comparison.baseline.value,
        candidate=comparison.candidate.value,
        delta=comparison.delta.value,
        ci_low=comparison.delta.ci_low,
        ci_high=comparison.delta.ci_high,
        correlation=comparison.correlation,
        units=units,
        default_margin=comparison.margin,
        grid=grid,
    )


FAMILY_PRIORITY = ("outcome", "safety", "trajectory", "rag", "judge", "efficiency")
"""Tie-break order. What an agent *achieved* is more interesting than what it spent."""


def panels_for(
    gate: GateResult, *, metrics: tuple[str, ...] = (), limit: int = 4
) -> list[MetricPanel]:
    """Build panels for the metrics that best demonstrate what the margin decides.

    Metrics are ranked by **how many distinct verdicts appear across the grid**, not by effect
    size. A slider whose entire travel yields one answer teaches nothing, however large the
    difference behind it; a metric that moves REGRESSION → UNDERPOWERED → PASS demonstrates the
    project's whole argument in one gesture. Ranking by |delta| instead reliably surfaces token
    counts, whose deltas are in the hundreds and whose verdicts never budge.

    Metrics whose comparison carries no analysis units are skipped rather than shown inert: a
    slider that cannot move is worse than an absent one, because it implies the answer is stable
    when in fact it was never computed.
    """
    candidates = [
        comparison
        for comparison in gate.comparison.metrics
        if len(comparison.analysis_units) >= 2 and (not metrics or comparison.metric in metrics)
    ]
    built = [build_panel(item, policy=gate.policy) for item in candidates]
    built.sort(key=_teaching_value, reverse=True)
    return built[:limit]


def _teaching_value(panel: MetricPanel) -> tuple[int, int, float]:
    """Rank key: verdict diversity, then family interest, then effect size."""
    distinct = len({point.verdict for point in panel.grid})
    family = panel.metric.split(".", 1)[0]
    rank = len(FAMILY_PRIORITY) - FAMILY_PRIORITY.index(family) if family in FAMILY_PRIORITY else 0
    return (distinct, rank, abs(panel.delta))


def panel_payload(panels: list[MetricPanel], *, scenario: str, suite: str) -> dict[str, Any]:
    """Serialise panels into the JSON the page reads."""
    return {
        "scenario": scenario,
        "suite": suite,
        "alphas": list(ALPHA_CHOICES),
        "margin_steps": MARGIN_STEPS,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "panels": [
            {
                "metric": panel.metric,
                "direction": panel.direction,
                "n_units": panel.n_units,
                "clustered": panel.clustered,
                "baseline": round(panel.baseline, 6),
                "candidate": round(panel.candidate, 6),
                "delta": round(panel.delta, 6),
                "ci_low": None if panel.ci_low is None else round(panel.ci_low, 6),
                "ci_high": None if panel.ci_high is None else round(panel.ci_high, 6),
                "correlation": (None if panel.correlation is None else round(panel.correlation, 4)),
                "default_margin": round(panel.default_margin, 6),
                "units": [round(value, 6) for value in panel.units],
                "grid": [
                    {
                        "m": round(point.margin, 6),
                        "a": point.alpha,
                        "v": point.verdict,
                        "pr": round(point.p_regression, 6),
                        "pn": round(point.p_noninferiority, 6),
                        "mde": None if point.mde is None else round(point.mde, 6),
                        "pw": None if point.power is None else round(point.power, 4),
                    }
                    for point in panel.grid
                ],
            }
            for panel in panels
        ],
    }


def write_payload(payload: dict[str, Any], target: Path) -> Path:
    """Write a payload to disk as compact JSON."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return target
