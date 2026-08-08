"""Self-contained HTML gate report (E4).

One file, no network, no CDN — it has to open from a CI artifact on a laptop with no internet
and still render its charts. Vega-Lite specs are embedded as JSON and drawn with inline SVG
rather than a charting library, which keeps the artifact small and the rendering deterministic.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from agentgate.gate.engine import GateResult, naive_verdict
from agentgate.report.charts import delta_chart_spec, pass_k_chart_spec, svg_delta_chart, svg_pass_k
from agentgate.report.markdown import format_delta, format_estimate, format_p
from agentgate.schemas.common import Verdict

BANNER_COLOURS: dict[Verdict, str] = {
    Verdict.PASS: "#1a7f37",
    Verdict.REGRESSION: "#cf222e",
    Verdict.SAFETY_FAIL: "#82071e",
    Verdict.UNDERPOWERED: "#9a6700",
    Verdict.INCONCLUSIVE: "#9a6700",
    Verdict.SKIPPED: "#57606a",
}

_TEMPLATE = """<!doctype html>
<html lang="en" data-verdict="{{ verdict.verdict.value }}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgentGate — {{ verdict.verdict.value }} — {{ verdict.suite.name }}</title>
<style>
  :root {
    --bg: #ffffff; --fg: #1f2328; --muted: #57606a; --line: #d0d7de;
    --panel: #f6f8fa; --accent: {{ banner_colour }};
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #0d1117; --fg: #e6edf3; --muted: #8b949e; --line: #30363d; --panel: #161b22; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg);
         font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif; }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
  .banner { background: var(--accent); color: #fff; border-radius: 10px;
            padding: 1.1rem 1.4rem; margin-bottom: 1.5rem; }
  .banner h1 { margin: 0 0 .35rem; font-size: 1.35rem; letter-spacing: .01em; }
  .banner p { margin: 0; opacity: .95; }
  .meta { color: var(--muted); font-size: .87rem; margin-bottom: 1.75rem; }
  .meta code { background: var(--panel); padding: .1rem .35rem; border-radius: 4px; }
  h2 { font-size: 1.05rem; margin: 2.25rem 0 .75rem; padding-bottom: .35rem;
       border-bottom: 1px solid var(--line); }
  .scroll { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-size: .88rem; }
  th, td { text-align: left; padding: .5rem .65rem; border-bottom: 1px solid var(--line);
           white-space: nowrap; }
  th { color: var(--muted); font-weight: 600; }
  td.metric { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; white-space: nowrap; }
  .pill { display: inline-block; padding: .1rem .5rem; border-radius: 999px;
          font-size: .76rem; font-weight: 600; color: #fff; }
  .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
           padding: 1rem 1.15rem; margin: .75rem 0; }
  .panel p:first-child { margin-top: 0; } .panel p:last-child { margin-bottom: 0; }
  ul { margin: .4rem 0 .4rem 1.1rem; padding: 0; }
  li { margin: .2rem 0; }
  .muted { color: var(--muted); }
  figure { margin: 1rem 0; }
  details { margin: .5rem 0; }
  summary { cursor: pointer; font-weight: 600; }
  footer { margin-top: 3rem; color: var(--muted); font-size: .8rem;
           border-top: 1px solid var(--line); padding-top: 1rem; }
</style>
</head>
<body><div class="wrap">

<div class="banner">
  <h1>{{ banner_text }}</h1>
  <p>{{ verdict.summary }}</p>
</div>

<p class="meta">
  Suite <code>{{ verdict.suite.name }}@{{ verdict.suite.version }}</code>
  · {{ verdict.suite.n_tasks }} tasks × K={{ comparison.k }}
  · {{ comparison.n_clusters or verdict.suite.n_tasks }} clusters
  · baseline <code>{{ verdict.baseline_run_id }}</code>
  · candidate <code>{{ verdict.candidate_run_id }}</code>
  · policy <code>{{ verdict.policy_hash[:10] }}</code>
  · generated {{ generated_at }}
</p>

<h2>Gated metrics</h2>
<div class="scroll">
<table>
  <thead><tr>
    <th>Verdict</th><th>Metric</th><th>Baseline</th><th>Candidate</th>
    <th>Δ (95% CI)</th><th>margin</th><th>p raw</th><th>p BH</th><th>power</th><th>test</th>
  </tr></thead>
  <tbody>
  {% for row in rows %}
    <tr>
      <td><span class="pill" style="background: {{ row.colour }}">{{ row.verdict }}</span></td>
      <td class="metric">{{ row.metric }}</td>
      <td>{{ row.baseline }}</td>
      <td>{{ row.candidate }}</td>
      <td>{{ row.delta }}</td>
      <td>{{ row.margin }}</td>
      <td>{{ row.p_raw }}</td>
      <td>{{ row.p_adj }}</td>
      <td>{{ row.power }}</td>
      <td class="muted">{{ row.test }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
</div>

<figure>{{ delta_svg | safe }}</figure>

{% if safety %}
<h2>Safety tripwires</h2>
<div class="panel">
  <p>Any <strong>new</strong> safety failure fails the gate with no hypothesis test in the way.</p>
  <div class="scroll"><table>
    <thead><tr><th>Metric</th><th>Task</th><th>Rep</th><th>Baseline</th><th>Candidate</th></tr></thead>
    <tbody>
    {% for finding in safety %}
      <tr><td class="metric">{{ finding.metric }}</td><td class="metric">{{ finding.task_id }}</td>
      <td>{{ finding.rep }}</td><td>{{ "FAIL" if finding.baseline_failed else "ok" }}</td>
      <td>{{ "FAIL" if finding.candidate_failed else "ok" }}</td></tr>
    {% endfor %}
    </tbody>
  </table></div>
</div>
{% endif %}

<h2>Naive threshold vs statistical gate</h2>
<div class="panel">
  <p>{{ naive_line }}</p>
  <p class="muted">The industry-standard rule &mdash; <em>fail if any metric drops more than
  3%</em> &mdash; never consults the sample size. On {{ verdict.suite.n_tasks }} tasks a 3-point
  move is well inside sampling noise.</p>
</div>

{% if pass_k_svg %}
<h2>Reliability: pass^k decay</h2>
<figure>{{ pass_k_svg | safe }}</figure>
<p class="muted">pass^k is the probability that <em>all</em> k repetitions succeed. A single-run
pass rate cannot show this curve, which is why every task runs K times (E2).</p>
{% if flakiest %}
<details><summary>Flakiest tasks (succeeded sometimes, not always)</summary>
<div class="scroll"><table>
  <thead><tr><th>Task</th><th>Successes</th><th>K</th></tr></thead>
  <tbody>
  {% for flake in flakiest %}
    <tr><td class="metric">{{ flake.task_id }}</td><td>{{ flake.successes }}</td>
    <td>{{ flake.k }}</td></tr>
  {% endfor %}
  </tbody>
</table></div>
</details>
{% endif %}
{% endif %}

<h2>Power and minimum detectable effect</h2>
<div class="panel"><ul>
{% for line in power_lines %}<li>{{ line }}</li>{% endfor %}
</ul></div>

{% if judge %}
<h2>Judge health</h2>
<div class="panel">
  <div class="scroll"><table><tbody>
  {% for label, value in judge %}<tr><th>{{ label }}</th><td>{{ value }}</td></tr>{% endfor %}
  </tbody></table></div>
</div>
{% endif %}

<h2>Why each metric ruled the way it did</h2>
{% for ruling in rulings %}
<details>
  <summary>{{ ruling.metric }} — {{ ruling.verdict.value }}</summary>
  <ul>{% for reason in ruling.reasons %}<li>{{ reason }}</li>{% endfor %}</ul>
</details>
{% endfor %}

{% if warnings %}
<h2>Warnings</h2>
<div class="panel"><ul>
{% for warning in warnings %}<li>{{ warning }}</li>{% endfor %}
</ul></div>
{% endif %}

<h2>Reproducibility</h2>
<div class="panel"><div class="scroll"><table><tbody>
{% for label, value in manifest_rows %}
  <tr><th>{{ label }}</th><td class="metric">{{ value }}</td></tr>
{% endfor %}
</tbody></table></div></div>

<footer>
  Generated by AgentGate. Every estimate on this page carries a 95% interval; every p-value is
  reported raw and Benjamini-Hochberg adjusted. Vega-Lite specs for these charts are embedded
  below for reuse.
  <script type="application/json" id="agentgate-charts">{{ chart_specs | safe }}</script>
</footer>

</div></body></html>
"""


def _environment() -> Environment:
    return Environment(
        autoescape=True, undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True
    )


def render_html(result: GateResult, *, generated_at: datetime | None = None) -> str:
    """Render the full HTML gate report.

    Args:
        result: The gate result.
        generated_at: Timestamp to display; defaults to now.

    Returns:
        A complete, self-contained HTML document.
    """
    verdict = result.verdict
    comparison = result.comparison
    colour = BANNER_COLOURS.get(verdict.verdict, "#57606a")

    rows = []
    for ruling in result.rulings:
        item = ruling.comparison
        rows.append(
            {
                "metric": ruling.metric,
                "verdict": ruling.verdict.value,
                "colour": BANNER_COLOURS.get(ruling.verdict, "#57606a"),
                "baseline": format_estimate(item.baseline) if item else "—",
                "candidate": format_estimate(item.candidate) if item else "—",
                "delta": format_delta(item) if item else "—",
                "margin": f"{ruling.margin:.3f}",
                "p_raw": format_p(ruling.p_regression),
                "p_adj": format_p(ruling.p_adjusted),
                "power": f"{ruling.achieved_power:.0%}"
                if ruling.achieved_power is not None
                else "—",
                "test": item.regression_test.test if item else "—",
            }
        )

    from agentgate.stats.power import describe

    power_lines = [
        describe(ruling.comparison.power, ruling.metric, ruling.margin)
        for ruling in result.rulings
        if ruling.comparison is not None and ruling.comparison.power is not None
    ] or ["No gated metric produced a power estimate."]

    judge_rows = []
    health = comparison.judge_health
    if health is not None:
        judge_rows = [
            ("judge model", health.judge_model or "—"),
            (
                "Cohen's kappa (vs human)",
                f"{health.cohens_kappa:.2f}"
                if health.cohens_kappa is not None
                else "not calibrated",
            ),
            (
                "Spearman rho",
                f"{health.spearman_rho:.2f}" if health.spearman_rho is not None else "—",
            ),
            (
                "position-flip rate",
                f"{health.position_flip_rate:.0%}"
                if health.position_flip_rate is not None
                else "—",
            ),
            (
                "verbosity correlation",
                f"{health.verbosity_correlation:+.2f}"
                if health.verbosity_correlation is not None
                else "—",
            ),
            (
                "judge drift",
                health.drift_detail or ("detected" if health.drift_detected else "none"),
            ),
            ("self-judging", "YES — scores inflated" if health.self_judging else "no"),
        ]

    naive = naive_verdict(comparison)
    gated_names = {ruling.metric for ruling in result.rulings}
    naive_fails = sorted(m for m, failed in naive.items() if failed and m in gated_names)
    naive_line = (
        f"A fixed −3% threshold gate would have FAILED on {', '.join(naive_fails)}."
        if naive_fails
        else "A fixed −3% threshold gate would have PASSED this comparison."
    )
    naive_line += f" AgentGate's verdict: {verdict.verdict.value}."

    reliability = comparison.reliability_candidate[0] if comparison.reliability_candidate else None
    pass_k_svg = ""
    if comparison.reliability_baseline and reliability:
        pass_k_svg = svg_pass_k(comparison.reliability_baseline[0], reliability)

    chart_specs = json.dumps(
        {
            "delta": delta_chart_spec(result),
            "pass_k": pass_k_chart_spec(comparison) if reliability else None,
        },
        indent=2,
    )

    manifest_rows = [
        ("comparison id", verdict.comparison_id),
        ("suite content hash", verdict.suite.content_hash),
        ("baseline run", verdict.baseline_run_id),
        ("candidate run", verdict.candidate_run_id),
        ("policy hash", verdict.policy_hash),
        ("exit code", str(verdict.exit_code)),
    ]

    return (
        _environment()
        .from_string(_TEMPLATE)
        .render(
            verdict=verdict,
            comparison=comparison,
            rulings=result.rulings,
            rows=rows,
            banner_colour=colour,
            banner_text=verdict.verdict.value.replace("_", " "),
            generated_at=(generated_at or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC"),
            safety=verdict.safety_failures,
            naive_line=naive_line,
            power_lines=power_lines,
            judge=judge_rows,
            warnings=[w for w in verdict.warnings if w],
            delta_svg=svg_delta_chart(result),
            pass_k_svg=pass_k_svg,
            flakiest=reliability.flakiest_tasks if reliability else [],
            chart_specs=chart_specs,
            manifest_rows=manifest_rows,
        )
    )


def write_html(
    result: GateResult, path: str | Path, *, generated_at: datetime | None = None
) -> Path:
    """Write the HTML report to ``path``.

    Args:
        result: The gate result.
        path: Destination file; parent directories are created.
        generated_at: Timestamp to display.

    Returns:
        The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(result, generated_at=generated_at), encoding="utf-8")
    return target
