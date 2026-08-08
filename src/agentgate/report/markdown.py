"""The sticky PR comment (E4).

Written for someone who has thirty seconds and a decision to make. The verdict banner answers
"can I merge?"; the table answers "what moved?"; the power note answers "should I believe the
answer?". Everything else lives in the HTML report behind a link.

Two rules the renderer never breaks:

* **No number without its interval.** A point estimate in a PR comment is how a 3-point wiggle
  becomes an argument.
* **The naive comparison is always shown.** Seeing what a fixed-threshold gate would have said
  on the same data is the fastest way to understand why this one exists.
"""

from __future__ import annotations

from agentgate.gate.engine import GateResult, MetricRuling, naive_verdict
from agentgate.schemas.common import Verdict
from agentgate.schemas.results import ComparisonResult, Estimate, MetricComparison

STICKY_MARKER = "<!-- agentgate-sticky-comment -->"
"""Marker the CI workflow greps for so it updates one comment instead of adding another."""

BANNERS: dict[Verdict, tuple[str, str]] = {
    Verdict.PASS: ("✅", "GATE PASSED"),
    Verdict.REGRESSION: ("❌", "GATE FAILED — STATISTICALLY SIGNIFICANT REGRESSION"),
    Verdict.SAFETY_FAIL: ("🛑", "GATE FAILED — SAFETY"),
    Verdict.UNDERPOWERED: ("⚠️", "UNDERPOWERED — THIS SUITE CANNOT TELL"),
    Verdict.INCONCLUSIVE: ("⚠️", "INCONCLUSIVE"),
    Verdict.SKIPPED: ("➖", "NOT EVALUATED"),
}

VERDICT_ICONS: dict[Verdict, str] = {
    Verdict.PASS: "✅",
    Verdict.REGRESSION: "❌",
    Verdict.INCONCLUSIVE: "⚠️",
    Verdict.UNDERPOWERED: "⚠️",
    Verdict.SAFETY_FAIL: "🛑",
    Verdict.SKIPPED: "➖",
}


def format_estimate(estimate: Estimate, *, digits: int = 3) -> str:
    """Render a value with its interval, or say plainly that there is no interval."""
    if estimate.ci_low is None or estimate.ci_high is None:
        return f"{estimate.value:.{digits}f} (no interval: {estimate.method})"
    return (
        f"{estimate.value:.{digits}f} [{estimate.ci_low:.{digits}f}, {estimate.ci_high:.{digits}f}]"
    )


def format_delta(comparison: MetricComparison, *, digits: int = 3) -> str:
    """Render the change with its interval and an explicit sign."""
    delta = comparison.delta
    sign = "+" if delta.value >= 0 else ""
    if delta.ci_low is None or delta.ci_high is None:
        return f"{sign}{delta.value:.{digits}f}"
    return (
        f"{sign}{delta.value:.{digits}f} [{delta.ci_low:+.{digits}f}, {delta.ci_high:+.{digits}f}]"
    )


def format_p(value: float | None) -> str:
    """Render a p-value, using scientific notation only where decimals stop being readable."""
    if value is None:
        return "—"
    if value < 1e-4:
        return f"{value:.1e}"
    return f"{value:.4f}"


def render_metric_table(result: GateResult) -> str:
    """The per-metric table: change, interval, p-values, power."""
    lines = [
        "| | Metric | Baseline | Candidate | Δ (95% CI) | margin | p (raw) | p (BH) | power |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for ruling in result.rulings:
        icon = VERDICT_ICONS.get(ruling.verdict, "•")
        comparison = ruling.comparison
        if comparison is None:
            lines.append(f"| {icon} | `{ruling.metric}` | — | — | — | — | — | — | — |")
            continue
        power = f"{ruling.achieved_power:.0%}" if ruling.achieved_power is not None else "—"
        lines.append(
            f"| {icon} | `{ruling.metric}` "
            f"| {format_estimate(comparison.baseline)} "
            f"| {format_estimate(comparison.candidate)} "
            f"| {format_delta(comparison)} "
            f"| {ruling.margin:.3f} "
            f"| {format_p(ruling.p_regression)} "
            f"| {format_p(ruling.p_adjusted)} "
            f"| {power} |"
        )
    return "\n".join(lines)


def render_pass_k_sparkline(comparison: ComparisonResult) -> str:
    """A text pass^k decay curve for baseline and candidate (E2)."""
    if not comparison.reliability_baseline or not comparison.reliability_candidate:
        return ""
    base = comparison.reliability_baseline[0]
    cand = comparison.reliability_candidate[0]
    ks = [point.k for point in base.curve]
    if not ks:
        return ""
    header = "| k | " + " | ".join(str(k) for k in ks) + " |"
    divider = "|---|" + "---|" * len(ks)
    base_row = (
        "| baseline | " + " | ".join(f"{point.pass_hat_k.value:.2f}" for point in base.curve) + " |"
    )
    cand_row = (
        "| candidate | "
        + " | ".join(f"{point.pass_hat_k.value:.2f}" for point in cand.curve)
        + " |"
    )
    return "\n".join(
        [
            "<details><summary>pass^k decay — the reliability number a single run hides</summary>",
            "",
            header,
            divider,
            base_row,
            cand_row,
            "",
            f"Flake rate: baseline {base.flake_rate.value:.0%}, candidate "
            f"{cand.flake_rate.value:.0%} (tasks that succeeded sometimes but not always).",
            "</details>",
        ]
    )


def render_safety(result: GateResult) -> str:
    """The safety section, present only when a tripwire fired."""
    findings = result.verdict.safety_failures
    if not findings:
        return ""
    lines = ["### 🛑 Safety tripwires", "", "| Metric | Task | Rep |", "|---|---|---|"]
    lines.extend(
        f"| `{finding.metric}` | `{finding.task_id}` | {finding.rep} |" for finding in findings
    )
    lines.append("")
    lines.append(
        "_Safety failures bypass the statistics entirely: any **new** failure fails the gate, "
        "however small the sample._"
    )
    return "\n".join(lines)


def render_naive_comparison(result: GateResult, *, threshold: float = 0.03) -> str:
    """The E6 argument: what a fixed-threshold gate would have concluded on this data."""
    naive = naive_verdict(result.comparison, threshold=threshold)
    gated = {ruling.metric for ruling in result.rulings}
    relevant = {metric: failed for metric, failed in naive.items() if metric in gated}
    if not relevant:
        return ""
    naive_fails = sorted(metric for metric, failed in relevant.items() if failed)
    ours = result.verdict.verdict.value

    if naive_fails and ours == "PASS":
        line = (
            f"A fixed `-{threshold:.0%}` threshold gate would have **failed** this PR on "
            f"{', '.join(f'`{m}`' for m in naive_fails)}. AgentGate passes it: the observed drop "
            f"is inside sampling noise at this sample size."
        )
    elif not naive_fails and ours in ("REGRESSION", "SAFETY_FAIL"):
        line = (
            f"A fixed `-{threshold:.0%}` threshold gate would have **passed** this PR. "
            f"AgentGate fails it: the change is small but statistically significant beyond the "
            f"configured margin."
        )
    elif not naive_fails and ours == "UNDERPOWERED":
        line = (
            f"A fixed `-{threshold:.0%}` threshold gate would have **passed** this PR silently. "
            f"AgentGate reports that the suite is too small to tell."
        )
    else:
        line = (
            f"A fixed `-{threshold:.0%}` threshold gate would have reached the same conclusion "
            f"here — but for the wrong reason: it never consults the sample size."
        )
    return f"### Naive threshold vs statistical gate\n\n{line}"


def render_power_note(result: GateResult) -> str:
    """The sentence C4 requires on every report."""
    from agentgate.stats.power import describe

    lines = []
    for ruling in result.rulings:
        if ruling.comparison is None or ruling.comparison.power is None:
            continue
        lines.append("- " + describe(ruling.comparison.power, ruling.metric, ruling.margin))
    if not lines:
        return ""
    return (
        "<details><summary>Power &amp; minimum detectable effect</summary>\n\n"
        + "\n".join(lines)
        + "\n</details>"
    )


def render_judge_panel(comparison: ComparisonResult) -> str:
    """Judge health, so a judge-backed number can be discounted appropriately (E4)."""
    health = comparison.judge_health
    if health is None:
        return ""
    rows = [
        ("judge model", health.judge_model or "—"),
        (
            "Cohen's κ (vs human)",
            f"{health.cohens_kappa:.2f}" if health.cohens_kappa else "not calibrated",
        ),
        ("Spearman ρ", f"{health.spearman_rho:.2f}" if health.spearman_rho else "—"),
        (
            "position-flip rate",
            f"{health.position_flip_rate:.0%}" if health.position_flip_rate is not None else "—",
        ),
        (
            "verbosity correlation",
            f"{health.verbosity_correlation:+.2f}"
            if health.verbosity_correlation is not None
            else "—",
        ),
        ("drift", "DETECTED" if health.drift_detected else "none"),
    ]
    body = "\n".join(f"| {label} | {value} |" for label, value in rows)
    return "<details><summary>Judge health</summary>\n\n| | |\n|---|---|\n" + body + "\n</details>"


def render_comment(result: GateResult, *, report_url: str = "") -> str:
    """Render the full sticky PR comment.

    Args:
        result: The gate result.
        report_url: Optional link to the uploaded HTML report.

    Returns:
        Markdown, prefixed with the sticky marker so CI updates one comment.
    """
    verdict = result.verdict
    icon, banner = BANNERS.get(verdict.verdict, ("•", verdict.verdict.value))
    comparison = result.comparison

    sections = [
        STICKY_MARKER,
        f"## {icon} {banner}",
        "",
        verdict.summary,
        "",
        f"**Suite** `{verdict.suite.name}@{verdict.suite.version}` · "
        f"{verdict.suite.n_tasks} tasks × K={comparison.k} · "
        f"{comparison.n_clusters or verdict.suite.n_tasks} clusters · "
        f"policy `{verdict.policy_hash[:8]}`",
        "",
        render_metric_table(result),
        "",
    ]

    for section in (
        render_safety(result),
        render_naive_comparison(result),
        render_pass_k_sparkline(comparison),
        render_power_note(result),
        render_judge_panel(comparison),
    ):
        if section:
            sections.extend([section, ""])

    if comparison.clustered:
        sections.append(
            "_Clustered standard errors are in use: this suite's tasks are grouped, so naive "
            "intervals would be too narrow._\n"
        )

    warnings = [warning for warning in verdict.warnings if warning]
    if warnings:
        sections.append("<details><summary>Warnings</summary>\n")
        sections.extend(f"- {warning}" for warning in warnings)
        sections.append("\n</details>\n")

    if report_url:
        sections.append(f"[Full HTML report]({report_url})")
    sections.append(
        f"\n<sub>baseline `{verdict.baseline_run_id}` · candidate `{verdict.candidate_run_id}` "
        f"· exit {verdict.exit_code}</sub>"
    )
    return "\n".join(sections).rstrip() + "\n"


def render_ruling_detail(ruling: MetricRuling) -> str:
    """Expanded reasoning for one metric, used by the HTML report and `agentgate compare -v`."""
    lines = [f"**{ruling.metric}** — {ruling.verdict.value}"]
    lines.extend(f"- {reason}" for reason in ruling.reasons)
    return "\n".join(lines)
