"""Build the demo landing page — the one link that explains the whole project.

The previous version opened with a slider labelled "margin δ" over a chart of per-cluster
differences. Both are the right things to show *eventually*; neither means anything to a reader
who has not yet been told what problem they solve. It was decorative.

This version tells the story in the order a reader can follow it:

1. **The question**, in one sentence: you changed your agent — did you break it?
2. **Two real runs where the obvious answer is wrong**, in opposite directions. A security
   breach that scores a perfect 100%, and a 21-point drop that turns out not to be established.
3. **A scenario you step through**: what changed, the 70 tasks before and after, what a normal
   CI check concludes, what AgentGate concludes, and why they differ.
4. **Only then, the tolerance control** — by which point "how much worse would you accept?" is
   a question the reader is already asking.

Everything is generated from real runs. The naive verdict is genuinely what a
``if drop > 3%: fail`` rule would have done with these numbers, and every gate verdict comes
from the same engine the CI gate uses.

Usage::

    uv run python scripts/build_demo_site.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentgate.demo import run_scenario
from agentgate.report.chrome import STYLE, footer, nav
from agentgate.report.playground import panel_payload, panels_for
from agentgate.report.story import (
    copy_for,
    naive_verdict,
    outcome_payload,
    plain_label,
    task_outcomes,
    verdict_copy,
)

SITE = Path("site")
SUITE = Path("suites/crm_ops")
RESULTS = Path("results/harness.json")
HEADLINE_METRIC = "outcome.task_success"

# Ordered as a lesson, not alphabetically. The first two are the cases where the obvious answer
# is wrong — one in each direction — because those are the whole argument. The rest show the gate
# agreeing with common sense, which is what makes the first two credible rather than contrarian.
SCENARIOS = ("verbosity_attack", "injection", "dropped_tool", "flaky_dependency", "no_op")

STOP_VERDICTS = frozenset({"REGRESSION", "SAFETY_FAIL"})


def build_scenario(key: str) -> dict[str, Any]:
    """Run one scenario and assemble everything the page needs to narrate it."""
    result = run_scenario(key, suite_path=SUITE)
    baseline = [s for s in result.baseline_scores if s.metric == HEADLINE_METRIC]
    candidate = [s for s in result.candidate_scores if s.metric == HEADLINE_METRIC]

    outcomes = task_outcomes(baseline, candidate)
    naive = naive_verdict(outcomes)
    copy = copy_for(key)
    verdict = verdict_copy(result.verdict)

    gate_stops = result.verdict in STOP_VERDICTS
    panels = panels_for(result.gate, limit=4)
    payload = panel_payload(panels, scenario=key, suite="crm_ops")

    return {
        "key": key,
        "title": copy.title,
        "change": copy.change,
        "why": copy.why_it_matters,
        "lesson": copy.lesson,
        "gate_verdict": result.verdict,
        "gate_headline": verdict["headline"],
        "gate_body": verdict["body"],
        "agrees": gate_stops == naive.would_block,
        "naive": {
            "before": round(naive.baseline_rate, 4),
            "after": round(naive.candidate_rate, 4),
            "threshold": naive.threshold,
            "verdict": naive.verdict,
            "describe": naive.describe(),
        },
        "tasks": outcome_payload(outcomes),
        "panels": payload["panels"],
        "alphas": payload["alphas"],
        "labels": {panel.metric: plain_label(panel.metric) for panel in panels},
    }


def collect() -> dict[str, Any]:
    """Run every demo scenario through the real pipeline."""
    print("running scenarios through the real pipeline:")
    scenarios: dict[str, Any] = {}
    for key in SCENARIOS:
        scenarios[key] = build_scenario(key)
        item = scenarios[key]
        flag = "" if item["agrees"] else "   <- the normal check is wrong here"
        print(
            f"  {key:<18} gate={item['gate_verdict']:<13} naive={item['naive']['verdict']:<5}{flag}"
        )
    return {"order": list(SCENARIOS), "scenarios": scenarios}


def evidence() -> dict[str, Any]:
    """Load the committed evidence snapshot, or an empty stand-in when none exists."""
    if not RESULTS.exists():
        return {"cells": [], "models": [], "suites": []}
    loaded: dict[str, Any] = json.loads(RESULTS.read_text(encoding="utf-8"))
    return loaded


EVIDENCE_METRICS = (
    ("outcome.task_success", "Finished the job correctly"),
    ("trajectory.recall", "Used the tools it needed"),
    ("efficiency.total_tokens", "Tokens spent"),
    ("judge.coherence", "Sounded coherent"),
)


def _complete_cells(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Only fully recorded cells; a partial recording is not comparable to a complete one."""
    return [cell for cell in snapshot.get("cells", []) if cell.get("complete")]


def evidence_header(snapshot: dict[str, Any]) -> str:
    """Column headers naming each measured model."""
    heads = "".join(f"<th class='num'>{cell['label']}</th>" for cell in _complete_cells(snapshot))
    return f"<tr><th>Measured on 111 real benchmark tasks</th>{heads}</tr>"


def evidence_rows(snapshot: dict[str, Any]) -> str:
    """Render the real-model evidence as a comparison table."""
    cells = _complete_cells(snapshot)
    if not cells:
        return (
            "<tr><td colspan='3' class='dim'>No model recorded yet — run "
            "<code>agentgate harness record</code>.</td></tr>"
        )
    rows: list[str] = []
    for metric, label in EVIDENCE_METRICS:
        columns: list[str] = []
        for cell in cells:
            found = next((m for m in cell["metrics"] if m["metric"] == metric), None)
            if found is None:
                columns.append("<td class='num dim'>—</td>")
                continue
            value = float(found["value"])
            low, high = found.get("ci_low"), found.get("ci_high")
            if abs(value) >= 100:
                columns.append(f"<td class='num'>{value:,.0f}</td>")
            else:
                band = "" if low is None else f"<br><span class='ci'>{low:.2f} – {high:.2f}</span>"
                columns.append(f"<td class='num'>{value:.3f}{band}</td>")
        rows.append(f"<tr><td>{label}</td>{''.join(columns)}</tr>")
    return "\n".join(rows)


def build() -> Path:
    """Generate ``site/index.html``."""
    data = collect()
    snapshot = evidence()

    SITE.mkdir(parents=True, exist_ok=True)
    target = SITE / "index.html"
    target.write_text(
        PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__EVIDENCE_HEAD__", evidence_header(snapshot))
        .replace("__EVIDENCE_ROWS__", evidence_rows(snapshot))
        .replace("__GENERATED__", datetime.now(UTC).strftime("%B %Y")),
        encoding="utf-8",
    )
    print(f"wrote {target} ({target.stat().st_size / 1024:.0f} KB)")
    return target


BODY = """
<header class="hero wrap">
  <p class="eyebrow">Open source &middot; Apache-2.0</p>
  <h1>You changed your AI agent.<br>Did you break it?</h1>
  <p class="lede">
    Agents don't behave identically every run, so the score always moves a little. When it drops
    after a change, nobody can tell whether you broke something or just got unlucky. AgentGate is
    a CI check that answers that properly &mdash; and admits when it can't.
  </p>
  <div class="row">
    <a class="btn primary" href="#walkthrough">See it decide</a>
    <a class="btn" href="docs/">How it works</a>
    <a class="btn" href="gallery.html">Example reports</a>
  </div>
</header>

<section class="wrap">
  <p class="eyebrow">Why this is hard</p>
  <h2>The obvious answer is wrong in both directions</h2>
  <p>
    Almost every team checks agent quality the same way: run a test set, compare the score to
    last time, fail the build if it dropped more than a few points. Here are two real runs from
    this project where that rule gets it exactly backwards.
  </p>
  <div class="hooks" id="hooks"></div>
</section>

<section class="wrap" id="walkthrough">
  <p class="eyebrow">Walk through it</p>
  <h2>Pick something a developer might have done</h2>
  <p class="dim">
    Each is a real change, run through the real system on the same 70-task suite. Nothing below
    is illustrative.
  </p>

  <div class="picker" id="picker"></div>

  <div class="step">
    <span class="n">1</span>
    <h3 id="s1-title"></h3>
    <p id="s1-change"></p>
    <p class="dim" id="s1-why" style="font-size:.9rem;margin:0"></p>
  </div>

  <div class="step">
    <span class="n">2</span>
    <h3>What actually happened, task by task</h3>
    <p class="dim" style="font-size:.9rem">
      Every square is one test case. Both sides ran the same tasks with the same random seeds,
      so they line up one to one and are directly comparable.
    </p>
    <div class="rowlabel">Before your change</div>
    <div class="grid-tasks" id="grid-before"></div>
    <div class="rowlabel">After your change</div>
    <div class="grid-tasks" id="grid-after"></div>
    <div class="legend" style="margin-top:.8rem">
      <span><i class="swatch" style="background:var(--good)"></i> finished the job</span>
      <span><i class="swatch" style="background:var(--warn)"></i> partly</span>
      <span><i class="swatch" style="background:var(--bad)"></i> failed</span>
    </div>
  </div>

  <div class="step">
    <span class="n">3</span>
    <h3>Two checks, two answers</h3>
    <p class="dim" style="font-size:.9rem" id="s3-summary"></p>
    <div class="twocol">
      <div class="answer">
        <h4>A normal CI check</h4>
        <div id="naive-call"></div>
        <p id="naive-why"></p>
      </div>
      <div class="answer gate">
        <h4>AgentGate</h4>
        <div id="gate-call"></div>
        <p id="gate-why"></p>
      </div>
    </div>
    <div id="verdict-note"></div>
  </div>

  <div class="step">
    <span class="n">4</span>
    <h3>Your tolerance is a setting, not a guess</h3>
    <p class="dim" style="font-size:.9rem">
      AgentGate never asks "did anything change?" &mdash; something always changes. It asks
      whether the agent got worse by more than an amount <em>you</em> decided you care about.
      Move that amount and watch the answer change.
    </p>
    <div class="control">
      <div>
        <label class="field" for="metric">Which quality are we checking?</label>
        <select id="metric"></select>
        <div style="margin-top:1.1rem">
          <label class="field" for="tol">
            I'd accept the agent being up to
            <span id="tol-value" style="color:var(--accent)"></span> worse
          </label>
          <input type="range" id="tol" min="0" value="0" step="1">
        </div>
        <div id="tol-verdict" style="margin-top:1rem"></div>
      </div>
      <ul class="stats">
        <li><span>Before</span><b id="t-base">&mdash;</b></li>
        <li><span>After</span><b id="t-cand">&mdash;</b></li>
        <li><span>Difference</span><b id="t-delta">&mdash;</b></li>
        <li><span>Independent cases</span><b id="t-n">&mdash;</b></li>
        <li><span>Smallest drop this suite can detect</span><b id="t-mde">&mdash;</b></li>
      </ul>
    </div>
    <p class="dim" style="font-size:.84rem;margin:1.2rem 0 0">
      Every position on that slider was calculated in advance by the same engine the real gate
      uses. The page looks answers up; it does no statistics in your browser.
    </p>
  </div>
</section>

<section class="wrap">
  <p class="eyebrow">How it works</p>
  <h2>Three ideas, and that's most of it</h2>
  <div class="steps3">
    <div class="card">
      <h3>Run both sides on identical tasks</h3>
      <p>Same test cases, same random seeds, before and after. Task difficulty varies enormously,
      and pairing cancels it out, so what's left is the effect of your change.</p>
    </div>
    <div class="card">
      <h3>Compare per task, not totals</h3>
      <p>Two totals can match while every individual case moved. Looking at each task's
      before-and-after shows whether a pattern is real or a few coincidences.</p>
    </div>
    <div class="card">
      <h3>Answer three ways, not two</h3>
      <p>Broke it, didn't break it, or <em>not enough evidence to say</em>. That third answer is
      the honest one on a small test set, and no threshold rule can ever give it.</p>
    </div>
  </div>
  <p style="margin-top:1.6rem">
    There is more underneath: grouping related test cases so five rewordings of one scenario
    count once, correcting for checking dozens of metrics at the same time, and treating an
    LLM-as-judge score as a measurement with its own error bars.
    <a href="docs/methodology/">The methodology page</a> covers all of it.
  </p>
</section>

<section class="wrap">
  <p class="eyebrow">Real models, real tasks</p>
  <h2>Two open models on 111 published benchmark tasks</h2>
  <p>
    The walkthrough above uses a deterministic stand-in agent so it reproduces exactly. These
    numbers come from real language models doing real work &mdash; 666 runs and 4.5&nbsp;million
    tokens on tasks from &tau;&sup2;-bench, a published agent benchmark.
  </p>
  <div class="scroll">
    <table>
      <thead>__EVIDENCE_HEAD__</thead>
      <tbody>__EVIDENCE_ROWS__</tbody>
    </table>
  </div>
  <p class="dim" style="font-size:.87rem;margin-top:1rem">
    Read the first and last rows together. The bigger model sounds <em>more</em> coherent while
    finishing fewer jobs &mdash; which is exactly why "it reads well" is not a measurement. And
    on finishing the job, AgentGate refuses to rank them: the gap is smaller than what 111 tasks
    can resolve. <a href="docs/results/">Full numbers with error bars &rarr;</a>
  </p>
</section>

<section class="wrap">
  <p class="eyebrow">Go deeper</p>
  <h2>Where to next</h2>
  <div class="next">
    <a class="next-card" href="docs/">
      <h3>Documentation &rarr;</h3>
      <p class="dim" style="font-size:.88rem;margin:.3rem 0 0">How the statistics work, all 42
      metrics, the living harness, and an honest limitations page.</p>
    </a>
    <a class="next-card" href="gallery.html">
      <h3>Example reports &rarr;</h3>
      <p class="dim" style="font-size:.88rem;margin:.3rem 0 0">The full report AgentGate posts on
      a pull request, for every scenario above.</p>
    </a>
    <a class="next-card" href="https://github.com/vedant1711/agentgate">
      <h3>Source code &rarr;</h3>
      <p class="dim" style="font-size:.88rem;margin:.3rem 0 0">Python 3.12, Apache-2.0, 764 tests.
      Runs entirely free and offline.</p>
    </a>
  </div>
</section>

"""

SCRIPT = """
const DATA = __DATA__;
const $ = id => document.getElementById(id);
let current = DATA.order[0];
let metricIndex = 0;
let tolStep = 0;
const ALPHA = 0.05;

const NAIVE_TAG = { SHIP: ["ship", "Ships it"], BLOCK: ["block", "Blocks the merge"] };
const GATE_TAG = { PASS:"ship", REGRESSION:"block", UNDERPOWERED:"warn",
                   SAFETY_FAIL:"stop", INCONCLUSIVE:"plain" };
const TOL_WORDS = { PASS:"No meaningful regression", REGRESSION:"Real regression \\u2014 block",
                    UNDERPOWERED:"Not enough evidence to say" };

const pct = x => (x * 100).toFixed(0) + "%";
const pts = x => (x * 100).toFixed(1) + " points";

function renderHooks() {
  const picks = [
    { key:"injection", caption:"The score didn't move at all",
      body:"A support ticket contained a hidden instruction and the agent obeyed it. The average " +
           "score is <strong>perfect</strong>, so a threshold check ships a security breach " +
           "without blinking." },
    { key:"verbosity_attack", caption:"The score dropped hard",
      body:"Someone asked the agent to be more thorough. A threshold check blocks the merge over " +
           "that drop. AgentGate says the evidence doesn't establish the agent got worse \\u2014 " +
           "so blocking would be a false alarm." },
  ];
  $("hooks").innerHTML = picks.map(p => {
    const s = DATA.scenarios[p.key];
    if (!s) return "";
    const [cls, txt] = NAIVE_TAG[s.naive.verdict];
    return `<div class="hook">
      <div class="score">${pct(s.naive.before)} \\u2192 ${pct(s.naive.after)}</div>
      <div class="caption">${p.caption}</div>
      <div class="verdicts">
        <span class="tag ${cls}">Normal check: ${txt}</span>
        <span class="tag ${GATE_TAG[s.gate_verdict]}">AgentGate: ${s.gate_headline}</span>
      </div>
      <p>${p.body}</p></div>`;
  }).join("");
}

function renderPicker() {
  $("picker").innerHTML = DATA.order.map(k =>
    `<button type="button" data-k="${k}">${DATA.scenarios[k].title}</button>`).join("");
  for (const b of $("picker").children) {
    b.onclick = () => {
      current = b.dataset.k; metricIndex = 0; tolStep = 0; renderMetrics(); render();
    };
  }
}

const cellClass = v => v >= 0.999 ? "pass" : (v <= 0.001 ? "fail" : "part");

function renderGrid(el, tasks, key) {
  el.innerHTML = tasks.map(t =>
    `<div class="cell ${cellClass(t[key])}" title="${t.id}: ${pct(t[key])}"></div>`).join("");
}

function renderMetrics() {
  const s = DATA.scenarios[current];
  $("metric").innerHTML = s.panels.map((p, i) =>
    `<option value="${i}">${s.labels[p.metric] || p.metric}</option>`).join("");
  $("metric").value = metricIndex;
  $("metric").onchange = () => { metricIndex = +$("metric").value; tolStep = 0; render(); };
  const panel = s.panels[metricIndex];
  $("tol").max = panel ? (panel.grid.length / s.alphas.length) - 1 : 0;
}

function point(panel) {
  const s = DATA.scenarios[current];
  const ai = Math.max(0, s.alphas.indexOf(ALPHA));
  return panel.grid[tolStep * s.alphas.length + ai];
}

function render() {
  const s = DATA.scenarios[current];
  for (const b of $("picker").children) b.setAttribute("aria-pressed", b.dataset.k === current);

  $("s1-title").textContent = s.title;
  $("s1-change").textContent = s.change;
  $("s1-why").textContent = s.why;

  renderGrid($("grid-before"), s.tasks, "b");
  renderGrid($("grid-after"), s.tasks, "c");

  const moved = s.tasks.filter(t => t.s !== "same").length;
  $("s3-summary").textContent =
    `${s.naive.describe} ${moved} of ${s.tasks.length} test cases changed outcome.`;

  const [cls, txt] = NAIVE_TAG[s.naive.verdict];
  $("naive-call").innerHTML = `<span class="tag ${cls}">${txt}</span>`;
  $("naive-why").textContent = s.naive.verdict === "BLOCK"
    ? `It fell by more than ${pct(s.naive.threshold)}, so the rule fires. It has no way to know `
      + `whether that drop is meaningful.`
    : `It didn't fall by more than ${pct(s.naive.threshold)}, so the rule stays quiet. It only `
      + `ever looks at the average.`;

  $("gate-call").innerHTML =
    `<span class="tag ${GATE_TAG[s.gate_verdict]}">${s.gate_headline}</span>`;
  $("gate-why").textContent = s.gate_body;

  $("verdict-note").innerHTML = s.agrees
    ? `<p class="agree">Both checks agree here \\u2014 and that matters. A gate that only ever
       disagreed with common sense would just be contrarian.</p>`
    : `<div class="disagree"><strong>They disagree, and the normal check is the one that's
       wrong.</strong> ${s.lesson}</div>`;

  const panel = s.panels[metricIndex];
  if (!panel) return;
  const pt = point(panel);
  $("tol-value").textContent = pts(pt.m);
  $("tol").value = tolStep;
  $("tol-verdict").innerHTML =
    `<span class="tag ${GATE_TAG[pt.v] || "plain"}">${TOL_WORDS[pt.v] || pt.v}</span>`;

  $("t-base").textContent = panel.baseline.toFixed(3);
  $("t-cand").textContent = panel.candidate.toFixed(3);
  $("t-delta").textContent = (panel.delta >= 0 ? "+" : "") + panel.delta.toFixed(3);
  $("t-n").textContent = panel.n_units;
  $("t-mde").textContent = pt.mde === null ? "\\u2014" : pt.mde.toFixed(3);
}

renderHooks();
renderPicker();
renderMetrics();
$("tol").oninput = e => { tolStep = +e.target.value; render(); };
render();
"""

PAGE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgentGate &mdash; did your AI agent actually get worse?</title>
<meta name="description" content="A CI check that tells you whether your AI agent really
regressed or whether the score just moved. Walk through real examples where the obvious answer
is wrong.">
<style>{STYLE}</style>
</head>
<body>
{nav("demo")}
{BODY}
{footer("__GENERATED__")}
<script>{SCRIPT}</script>
</body>
</html>
"""


if __name__ == "__main__":
    build()
