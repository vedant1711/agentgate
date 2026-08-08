"""Charts for the HTML report: inline SVG for rendering, Vega-Lite specs for reuse.

Two representations of the same data, deliberately. The SVG renders with no JavaScript at all —
a CI artifact opened offline must still show its charts. The Vega-Lite spec is embedded beside
it so the numbers can be lifted into a notebook or a dashboard without re-deriving them.

Both charts show intervals rather than points. A bare bar chart of "before vs after" is the
visual form of the mistake this project exists to correct.
"""

from __future__ import annotations

from typing import Any

from agentgate.gate.engine import GateResult
from agentgate.schemas.results import ComparisonResult, ReliabilityReport

WIDTH = 720
ROW_HEIGHT = 34
PADDING = 16
LABEL_WIDTH = 230

COLOURS = {
    "PASS": "#1a7f37",
    "REGRESSION": "#cf222e",
    "SAFETY_FAIL": "#82071e",
    "UNDERPOWERED": "#9a6700",
    "INCONCLUSIVE": "#9a6700",
    "SKIPPED": "#8b949e",
}


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def svg_delta_chart(result: GateResult) -> str:
    """A forest plot of per-metric change with 95% intervals and the margin line.

    The reader's eye goes to whether an interval crosses the margin, which is exactly the
    question the gate answers — so the chart and the verdict cannot tell different stories.
    """
    rulings = [ruling for ruling in result.rulings if ruling.comparison is not None]
    if not rulings:
        return '<p class="muted">No gated metric produced a comparison to chart.</p>'

    bounds = []
    for ruling in rulings:
        delta = ruling.comparison.delta  # type: ignore[union-attr]
        bounds.extend(
            [
                delta.ci_low if delta.ci_low is not None else delta.value,
                delta.ci_high if delta.ci_high is not None else delta.value,
                -ruling.margin,
                ruling.margin,
            ]
        )
    low, high = min(bounds), max(bounds)
    span = max(high - low, 1e-9) * 1.2
    centre = (high + low) / 2.0
    low, high = centre - span / 2, centre + span / 2

    plot_left = LABEL_WIDTH + PADDING
    plot_width = WIDTH - plot_left - PADDING * 2
    height = PADDING * 2 + ROW_HEIGHT * len(rulings) + 28

    def x_of(value: float) -> float:
        return plot_left + (value - low) / (high - low) * plot_width

    parts = [
        f'<svg viewBox="0 0 {WIDTH} {height}" width="100%" role="img" '
        f'aria-label="Per-metric change with 95% confidence intervals" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="system-ui, sans-serif">',
        f'<line x1="{x_of(0):.1f}" y1="{PADDING}" x2="{x_of(0):.1f}" y2="{height - 28:.1f}" '
        f'stroke="currentColor" stroke-opacity="0.35" stroke-dasharray="3 3"/>',
    ]

    for index, ruling in enumerate(rulings):
        comparison = ruling.comparison
        assert comparison is not None
        delta = comparison.delta
        y = PADDING + ROW_HEIGHT * index + ROW_HEIGHT / 2
        colour = COLOURS.get(ruling.verdict.value, "#57606a")
        ci_low = delta.ci_low if delta.ci_low is not None else delta.value
        ci_high = delta.ci_high if delta.ci_high is not None else delta.value

        parts.append(
            f'<text x="{PADDING}" y="{y + 4:.1f}" font-size="12" fill="currentColor">'
            f"{_escape(ruling.metric)}</text>"
        )
        margin_x = x_of(-ruling.margin)
        parts.append(
            f'<line x1="{margin_x:.1f}" y1="{y - 12:.1f}" x2="{margin_x:.1f}" y2="{y + 12:.1f}" '
            f'stroke="#cf222e" stroke-opacity="0.5" stroke-width="2"/>'
        )
        parts.append(
            f'<line x1="{x_of(ci_low):.1f}" y1="{y:.1f}" x2="{x_of(ci_high):.1f}" y2="{y:.1f}" '
            f'stroke="{colour}" stroke-width="2.5" stroke-linecap="round"/>'
        )
        parts.append(f'<circle cx="{x_of(delta.value):.1f}" cy="{y:.1f}" r="4.5" fill="{colour}"/>')
        parts.append(
            f"<title>{_escape(ruling.metric)}: {delta.value:+.4f} "
            f"[{ci_low:+.4f}, {ci_high:+.4f}], margin -{ruling.margin:.4f}</title>"
        )

    baseline_y = height - 14
    for value in (low, centre, high):
        parts.append(
            f'<text x="{x_of(value):.1f}" y="{baseline_y}" font-size="10" '
            f'text-anchor="middle" fill="currentColor" opacity="0.6">{value:+.3f}</text>'
        )
    parts.append(
        f'<text x="{plot_left}" y="{PADDING - 4}" font-size="10" fill="currentColor" '
        f'opacity="0.6">change (candidate − baseline), 95% CI; red tick = margin</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def svg_pass_k(baseline: ReliabilityReport, candidate: ReliabilityReport) -> str:
    """The pass^k decay curve for both systems (E2)."""
    ks = [point.k for point in baseline.curve]
    if not ks:
        return ""
    height = 240
    left, right, top, bottom = 44, WIDTH - 20, 24, height - 34
    span_x = max(len(ks) - 1, 1)

    def x_of(index: int) -> float:
        return left + index / span_x * (right - left)

    def y_of(value: float) -> float:
        return bottom - value * (bottom - top)

    parts = [
        f'<svg viewBox="0 0 {WIDTH} {height}" width="100%" role="img" '
        f'aria-label="pass^k decay curve" xmlns="http://www.w3.org/2000/svg" '
        f'font-family="system-ui, sans-serif">',
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_of(fraction)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="currentColor" '
            f'stroke-opacity="0.12"/>'
            f'<text x="{left - 8}" y="{y + 3:.1f}" font-size="10" text-anchor="end" '
            f'fill="currentColor" opacity="0.6">{fraction:.2f}</text>'
        )

    for report, colour, label in (
        (baseline, "#57606a", "baseline"),
        (candidate, "#0969da", "candidate"),
    ):
        points = " ".join(
            f"{x_of(i):.1f},{y_of(point.pass_hat_k.value):.1f}"
            for i, point in enumerate(report.curve)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2.5"/>'
        )
        for i, point in enumerate(report.curve):
            parts.append(
                f'<circle cx="{x_of(i):.1f}" cy="{y_of(point.pass_hat_k.value):.1f}" r="3.5" '
                f'fill="{colour}"><title>{label} pass^{point.k} = '
                f"{point.pass_hat_k.value:.3f}</title></circle>"
            )

    for index, k in enumerate(ks):
        parts.append(
            f'<text x="{x_of(index):.1f}" y="{bottom + 16:.1f}" font-size="11" '
            f'text-anchor="middle" fill="currentColor" opacity="0.7">k={k}</text>'
        )
    parts.append(
        f'<text x="{left}" y="{top - 8}" font-size="10" fill="currentColor" opacity="0.6">'
        f"P(all k repetitions succeed) &mdash; grey: baseline, blue: candidate</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def delta_chart_spec(result: GateResult) -> dict[str, Any]:
    """Vega-Lite spec for the per-metric change chart."""
    values = [
        {
            "metric": ruling.metric,
            "delta": ruling.comparison.delta.value,
            "ci_low": ruling.comparison.delta.ci_low,
            "ci_high": ruling.comparison.delta.ci_high,
            "margin": -ruling.margin,
            "verdict": ruling.verdict.value,
        }
        for ruling in result.rulings
        if ruling.comparison is not None
    ]
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Per-metric change with 95% confidence intervals and the gate margin.",
        "data": {"values": values},
        "encoding": {"y": {"field": "metric", "type": "nominal", "title": None}},
        "layer": [
            {
                "mark": {"type": "rule", "strokeWidth": 2.5},
                "encoding": {
                    "x": {"field": "ci_low", "type": "quantitative", "title": "change"},
                    "x2": {"field": "ci_high"},
                    "color": {"field": "verdict", "type": "nominal"},
                },
            },
            {
                "mark": {"type": "point", "filled": True, "size": 70},
                "encoding": {
                    "x": {"field": "delta", "type": "quantitative"},
                    "color": {"field": "verdict", "type": "nominal"},
                },
            },
            {
                "mark": {"type": "tick", "color": "#cf222e", "thickness": 2},
                "encoding": {"x": {"field": "margin", "type": "quantitative"}},
            },
        ],
    }


def pass_k_chart_spec(comparison: ComparisonResult) -> dict[str, Any]:
    """Vega-Lite spec for the pass^k decay curve."""
    values: list[dict[str, Any]] = []
    for label, reports in (
        ("baseline", comparison.reliability_baseline),
        ("candidate", comparison.reliability_candidate),
    ):
        for report in reports:
            values.extend(
                {
                    "system": label,
                    "k": point.k,
                    "pass_hat_k": point.pass_hat_k.value,
                    "ci_low": point.pass_hat_k.ci_low,
                    "ci_high": point.pass_hat_k.ci_high,
                }
                for point in report.curve
            )
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "pass^k decay: probability that all k repetitions succeed.",
        "data": {"values": values},
        "mark": {"type": "line", "point": True},
        "encoding": {
            "x": {"field": "k", "type": "ordinal", "title": "k"},
            "y": {
                "field": "pass_hat_k",
                "type": "quantitative",
                "scale": {"domain": [0, 1]},
                "title": "pass^k",
            },
            "color": {"field": "system", "type": "nominal"},
        },
    }
