"""Reporting (L7): sticky PR comments, self-contained HTML reports, and charts."""

from agentgate.report.charts import (
    delta_chart_spec,
    pass_k_chart_spec,
    svg_delta_chart,
    svg_pass_k,
)
from agentgate.report.html import render_html, write_html
from agentgate.report.markdown import (
    STICKY_MARKER,
    format_delta,
    format_estimate,
    format_p,
    render_comment,
    render_metric_table,
    render_naive_comparison,
)

__all__ = [
    "STICKY_MARKER",
    "delta_chart_spec",
    "format_delta",
    "format_estimate",
    "format_p",
    "pass_k_chart_spec",
    "render_comment",
    "render_html",
    "render_metric_table",
    "render_naive_comparison",
    "svg_delta_chart",
    "svg_pass_k",
    "write_html",
]
