"""Build the interactive demo landing page for GitHub Pages.

This is the page a reader lands on. Its job is to make one idea tangible in a single gesture:
**a verdict is a function of the margin and the sample size, not a property of the numbers alone.**
Drag the margin slider and REGRESSION becomes UNDERPOWERED becomes PASS on real data.

Nothing here is illustrative. The scenarios are run through the actual pipeline, the differences
are the actual per-cluster paired differences, and every verdict on every slider position was
computed at build time by the same engine the gate uses. The page looks values up; it does not
compute statistics.

Usage::

    uv run python scripts/build_demo_site.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentgate.demo import run_scenario
from agentgate.report.playground import panel_payload, panels_for

SITE = Path("site")
SUITE = Path("suites/crm_ops")
RESULTS = Path("results/harness.json")

# One scenario per verdict the gate can reach, so the dropdown demonstrates the whole rule
# rather than one happy path: nothing changed, a tool vanished, output got wordier without
# getting worse, and a prompt-injection payload leaked. Four scenarios, four different verdicts.
SCENARIOS = ("no_op", "dropped_tool", "verbosity_attack", "injection")

VERDICT_BLURB = {
    "PASS": "Non-inferiority was <em>established</em> — not merely unrefuted.",
    "REGRESSION": "A one-sided test rejected beyond the margin. Block the merge.",
    "UNDERPOWERED": "The suite cannot tell. That is the honest answer, not a default to green.",
    "SAFETY_FAIL": "A safety tripwire fired. Tripwires bypass the statistics entirely.",
    "INCONCLUSIVE": "No metric produced a rulable comparison.",
}


def collect() -> dict[str, Any]:
    """Run every demo scenario and collect its interactive payload."""
    payloads: dict[str, Any] = {}
    for scenario in SCENARIOS:
        result = run_scenario(scenario, suite_path=SUITE)
        payload = panel_payload(
            panels_for(result.gate, limit=4), scenario=scenario, suite="crm_ops"
        )
        payload["gate_verdict"] = result.verdict
        payload["reason"] = result.gate.verdict.summary
        payloads[scenario] = payload
        print(f"  {scenario:<16} {result.verdict:<14} {len(payload['panels'])} panels")
    return payloads


def evidence() -> dict[str, Any]:
    """Load the committed evidence snapshot, or an empty stand-in when none exists."""
    if not RESULTS.exists():
        return {"cells": [], "models": [], "suites": []}
    loaded: dict[str, Any] = json.loads(RESULTS.read_text(encoding="utf-8"))
    return loaded


HEADLINE = (
    "outcome.task_success",
    "judge.coherence",
    "trajectory.recall",
    "trajectory.argument_correctness",
)


def evidence_rows(snapshot: dict[str, Any]) -> str:
    """Render the real-model evidence as table rows."""
    rows: list[str] = []
    for cell in snapshot.get("cells", []):
        by_name = {metric["metric"]: metric for metric in cell["metrics"]}
        for name in HEADLINE:
            metric = by_name.get(name)
            if metric is None:
                continue
            low, high = metric.get("ci_low"), metric.get("ci_high")
            interval = "—" if low is None else f"[{low:.3f}, {high:.3f}]"
            rows.append(
                f"<tr><td>{cell['label']}</td><td><code>{name}</code></td>"
                f"<td class='num'>{metric['value']:.3f}</td>"
                f"<td class='num dim'>{interval}</td>"
                f"<td class='num dim'>{metric['n']}</td></tr>"
            )
    if not rows:
        return (
            "<tr><td colspan='5' class='dim'>No model recorded yet. "
            "Run <code>agentgate harness record</code>.</td></tr>"
        )
    return "\n".join(rows)


def build() -> Path:
    """Generate ``site/index.html``."""
    print("running demo scenarios through the real pipeline:")
    payloads = collect()
    snapshot = evidence()

    SITE.mkdir(parents=True, exist_ok=True)
    target = SITE / "index.html"
    target.write_text(
        PAGE.replace("__DATA__", json.dumps(payloads, separators=(",", ":")))
        .replace("__BLURB__", json.dumps(VERDICT_BLURB, separators=(",", ":")))
        .replace("__EVIDENCE_ROWS__", evidence_rows(snapshot))
        .replace("__N_MODELS__", str(len(snapshot.get("models", []))))
        .replace("__GENERATED__", datetime.now(UTC).strftime("%Y-%m-%d")),
        encoding="utf-8",
    )
    size = target.stat().st_size
    print(f"wrote {target} ({size / 1024:.0f} KB)")
    return target


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgentGate — a statistical regression gate for AI agents</title>
<meta name="description" content="A CI check that blocks a pull request only when an agent has
statistically significantly regressed. Drag the margin and watch the verdict change on real data.">
<style>
  :root {
    --paper:#fbfaf7; --ink:#10131a; --dim:#5b6472; --line:#dcd8d0; --panel:#f2efe9;
    --accent:#0f766e; --accent-soft:#d6ece9; --on-accent:#ffffff;
    --pass:#067647; --regression:#b42318; --underpowered:#b54708; --safety:#6941c6;
    color-scheme: light;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --paper:#0e1116; --ink:#e8eaed; --dim:#98a2b3; --line:#252b34; --panel:#161b22;
      --accent:#5eead4; --accent-soft:#0d2b28; --on-accent:#0e1116;
      --pass:#5bd99a; --regression:#ff8a80; --underpowered:#fdb022; --safety:#c3b5fd;
      color-scheme: dark;
    }
  }
  :root[data-theme="dark"] {
    --paper:#0e1116; --ink:#e8eaed; --dim:#98a2b3; --line:#252b34; --panel:#161b22;
    --accent:#5eead4; --accent-soft:#0d2b28; --on-accent:#0e1116;
    --pass:#5bd99a; --regression:#ff8a80; --underpowered:#fdb022; --safety:#c3b5fd;
    color-scheme: dark;
  }

  * { box-sizing:border-box; }
  html { -webkit-text-size-adjust:100%; }
  body {
    margin:0; background:var(--paper); color:var(--ink);
    font:16px/1.65 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    font-variant-numeric:tabular-nums;
  }
  .wrap { max-width:940px; margin:0 auto; padding:0 1.25rem; }
  section { padding:3.5rem 0; border-top:1px solid var(--line); }
  section:first-of-type { border-top:0; }

  h1,h2,h3 { font-family:ui-serif,Georgia,"Times New Roman",serif; font-weight:600;
    text-wrap:balance; letter-spacing:-0.015em; margin:0 0 .6rem; }
  h1 { font-size:clamp(2.1rem,5.5vw,3.4rem); line-height:1.08; }
  h2 { font-size:clamp(1.4rem,3vw,1.9rem); margin-bottom:1rem; }
  h3 { font-size:1.05rem; font-family:inherit; font-weight:650; }
  p { margin:0 0 1rem; max-width:66ch; }
  .dim { color:var(--dim); }
  code, .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.88em; }
  a { color:var(--accent); text-underline-offset:.18em; }
  .eyebrow { font-size:.72rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase;
    color:var(--accent); margin:0 0 .9rem; }

  /* hero */
  header { padding:4.5rem 0 3.5rem; }
  .lede { font-size:clamp(1.05rem,2.1vw,1.28rem); color:var(--dim); max-width:60ch;
    margin-bottom:1.6rem; }
  .verdicts { display:flex; flex-wrap:wrap; gap:.5rem; margin:1.6rem 0 0; }
  .links { display:flex; flex-wrap:wrap; gap:.7rem; margin-top:2rem; }
  .btn { display:inline-block; padding:.6rem 1.1rem; border-radius:6px; text-decoration:none;
    border:1px solid var(--line); color:var(--ink); font-size:.92rem; font-weight:550; }
  .btn.primary { background:var(--accent); border-color:var(--accent);
    color:var(--on-accent); }
  .btn:hover { border-color:var(--accent); }

  .chip { display:inline-flex; align-items:center; gap:.4rem; padding:.22rem .6rem;
    border-radius:999px; font-size:.72rem; font-weight:700; letter-spacing:.05em;
    border:1px solid currentColor; }
  .chip::before { content:""; width:.45rem; height:.45rem; border-radius:50%;
    background:currentColor; }
  .v-PASS{color:var(--pass);} .v-REGRESSION{color:var(--regression);}
  .v-UNDERPOWERED{color:var(--underpowered);} .v-SAFETY_FAIL{color:var(--safety);}
  .v-INCONCLUSIVE{color:var(--dim);}

  /* instrument */
  .instrument { border:1px solid var(--line); border-radius:12px; background:var(--panel);
    overflow:hidden; }
  .controls { display:grid; gap:1.1rem; padding:1.3rem; border-bottom:1px solid var(--line);
    grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); align-items:end; }
  label.field { display:block; font-size:.7rem; font-weight:700; letter-spacing:.09em;
    text-transform:uppercase; color:var(--dim); margin-bottom:.4rem; }
  select, input[type=range] { width:100%; }
  select { padding:.45rem .5rem; border-radius:6px; border:1px solid var(--line);
    background:var(--paper); color:var(--ink); font:inherit; font-size:.9rem; }
  input[type=range] { accent-color:var(--accent); }
  .radios { display:flex; gap:.4rem; }
  .radios button { flex:1; padding:.42rem 0; border-radius:6px; border:1px solid var(--line);
    background:var(--paper); color:var(--dim); font:inherit; font-size:.85rem; cursor:pointer; }
  .radios button[aria-pressed=true] { border-color:var(--accent); color:var(--accent);
    background:var(--accent-soft); font-weight:650; }

  .readout { padding:1.4rem 1.3rem; display:grid; gap:1.3rem;
    grid-template-columns:minmax(0,1fr) minmax(0,260px); }
  @media (max-width:720px){ .readout{grid-template-columns:1fr;} }
  .verdict-line { display:flex; align-items:center; gap:.7rem; flex-wrap:wrap;
    margin-bottom:.5rem; }
  .verdict-line .chip { font-size:.8rem; padding:.3rem .75rem; }
  .stats { list-style:none; margin:0; padding:0; font-size:.85rem; }
  .stats li { display:flex; justify-content:space-between; gap:1rem; padding:.32rem 0;
    border-bottom:1px dotted var(--line); }
  .stats li:last-child{border-bottom:0;}
  .stats b { font-weight:600; font-family:ui-monospace,monospace; }

  figure { margin:0; }
  figcaption { font-size:.76rem; color:var(--dim); margin-top:.5rem; }
  .plot { width:100%; height:150px; display:block; }

  table { width:100%; border-collapse:collapse; font-size:.87rem; }
  th,td { text-align:left; padding:.5rem .6rem; border-bottom:1px solid var(--line); }
  th { font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; color:var(--dim); }
  td.num { text-align:right; font-family:ui-monospace,monospace; }
  .scroll { overflow-x:auto; }

  .cards { display:grid; gap:1rem; grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
    margin-top:1.4rem; }
  .card { border:1px solid var(--line); border-radius:10px; padding:1.1rem;
    background:var(--panel); }
  .card h3 { margin-bottom:.35rem; }
  .card p { font-size:.88rem; margin:0; color:var(--dim); }

  footer { padding:3rem 0 4rem; border-top:1px solid var(--line); font-size:.85rem;
    color:var(--dim); }
  .note { border-left:3px solid var(--accent); padding:.2rem 0 .2rem 1rem; margin:1.5rem 0;
    color:var(--dim); font-size:.92rem; }
  svg text { font-family:ui-sans-serif,-apple-system,sans-serif; }
</style>
</head>
<body>

<header class="wrap">
  <p class="eyebrow">Open source · Apache-2.0</p>
  <h1>Your agent scored 0.68 this week and 0.72 last week.<br>Did it regress?</h1>
  <p class="lede">
    Usually nobody knows. Agents are stochastic, so scores move on their own, and a dashboard
    cannot tell a regression from a Tuesday. AgentGate is a CI check that blocks a pull request
    only when the difference is <strong>larger than the noise</strong> — and says so out loud when
    it cannot tell.
  </p>
  <div class="verdicts">
    <span class="chip v-REGRESSION">REGRESSION</span>
    <span class="chip v-PASS">PASS</span>
    <span class="chip v-UNDERPOWERED">UNDERPOWERED</span>
  </div>
  <div class="links">
    <a class="btn primary" href="#try">Try the gate</a>
    <a class="btn" href="docs/">Documentation</a>
    <a class="btn" href="gallery.html">Report gallery</a>
    <a class="btn" href="https://github.com/vedant1711/agentgate">GitHub</a>
  </div>
</header>

<section class="wrap" id="try">
  <p class="eyebrow">Interactive</p>
  <h2>Move the margin. Watch the verdict change.</h2>
  <p>
    Below is a real comparison: a healthy baseline agent against a deliberately broken candidate,
    both run over the same 70-task suite with the same seeds. The dots are the actual per-cluster
    paired differences. The line is the non-inferiority margin.
  </p>
  <p class="dim" style="font-size:.9rem">
    Every verdict here was computed at build time by the same engine the gate uses — this page
    looks values up, it does not do statistics in your browser.
  </p>

  <div class="instrument">
    <div class="controls">
      <div>
        <label class="field" for="scenario">What changed</label>
        <select id="scenario"></select>
      </div>
      <div>
        <label class="field" for="metric">Metric</label>
        <select id="metric"></select>
      </div>
      <div>
        <label class="field" for="margin">Margin &delta; — <span id="margin-value"
          class="mono"></span></label>
        <input type="range" id="margin" min="0" value="0" step="1">
      </div>
      <div>
        <label class="field">Significance &alpha;</label>
        <div class="radios" id="alphas"></div>
      </div>
    </div>

    <div class="readout">
      <div>
        <div class="verdict-line">
          <span class="chip" id="verdict">—</span>
          <span class="dim" id="verdict-note" style="font-size:.87rem"></span>
        </div>
        <figure>
          <svg class="plot" id="plot" viewBox="0 0 620 150"
               preserveAspectRatio="xMidYMid meet" role="img"
               aria-label="Per-cluster paired differences against the margin"></svg>
          <figcaption id="plot-caption"></figcaption>
        </figure>
      </div>
      <ul class="stats">
        <li><span>Baseline</span><b id="s-base">—</b></li>
        <li><span>Candidate</span><b id="s-cand">—</b></li>
        <li><span>Difference</span><b id="s-delta">—</b></li>
        <li><span>95% CI</span><b id="s-ci">—</b></li>
        <li><span>p (regression)</span><b id="s-pr">—</b></li>
        <li><span>p (non-inferiority)</span><b id="s-pn">—</b></li>
        <li><span>Detectable effect</span><b id="s-mde">—</b></li>
        <li><span>Analysis units</span><b id="s-n">—</b></li>
      </ul>
    </div>
  </div>

  <div class="note">
    <strong>Try this:</strong> set the margin to zero. Almost everything becomes UNDERPOWERED —
    proving <em>exact</em> equality needs far more data than anyone has. That is why the gate asks
    a narrower question: is the candidate worse by more than an amount you declared you care about?
  </div>
</section>

<section class="wrap">
  <p class="eyebrow">Why it is trustworthy</p>
  <h2>Four decisions do most of the work</h2>
  <div class="cards">
    <div class="card">
      <h3>Pairing</h3>
      <p>Both systems run identical task instances with identical seeds. Task difficulty varies
      enormously and that variance dominates any unpaired comparison — pairing cancels it.</p>
    </div>
    <div class="card">
      <h3>Clustering</h3>
      <p>Five paraphrases of one scenario are one fact, not five. Counting them as five inflates
      the sample size and makes the gate fire on noise.</p>
    </div>
    <div class="card">
      <h3>Multiplicity</h3>
      <p>Gating 40 metrics at &alpha;=0.05 false-alarms ~87% of the time by chance alone.
      Benjamini–Hochberg controls the false discovery rate across the family.</p>
    </div>
    <div class="card">
      <h3>Judge variance</h3>
      <p>When a score comes from an LLM judge, the judge's own variance is propagated into the
      standard error. A judged number with hidden variance is confidence without basis.</p>
    </div>
  </div>
</section>

<section class="wrap">
  <p class="eyebrow">Architecture</p>
  <h2>Recording takes hours. Deciding takes seconds.</h2>
  <p>
    That gap is the whole design. The provider layer records a model's responses once and replays
    them forever, so CI never calls a model at all — which also makes every verdict reproducible
    byte-for-byte.
  </p>
  <div class="scroll">
  <svg viewBox="0 0 900 260" width="100%" style="max-width:900px;height:auto" role="img"
       aria-label="Pipeline: recording, scoring, deciding">
    <defs>
      <marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
              orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker>
    </defs>
    <g fill="none" stroke="currentColor" stroke-width="1.4" opacity=".45"
       marker-end="url(#a)">
      <path d="M215 70 H275"/><path d="M475 70 H535"/>
      <path d="M375 108 V150"/><path d="M375 188 H535"/>
      <path d="M700 108 V150"/><path d="M700 188 H745" />
    </g>
    <g font-size="13">
      <rect x="30" y="42" width="185" height="56" rx="8" fill="var(--panel)"
            stroke="var(--line)"/>
      <text x="122" y="66" text-anchor="middle" font-weight="650">Suite</text>
      <text x="122" y="84" text-anchor="middle" font-size="11" fill="var(--dim)">
        tasks + gold trajectories</text>

      <rect x="275" y="42" width="200" height="56" rx="8" fill="var(--panel)"
            stroke="var(--line)"/>
      <text x="375" y="66" text-anchor="middle" font-weight="650">Runner + agent</text>
      <text x="375" y="84" text-anchor="middle" font-size="11" fill="var(--dim)">
        fixed seeds, resumable</text>

      <rect x="535" y="42" width="200" height="56" rx="8" fill="var(--panel)"
            stroke="var(--line)"/>
      <text x="635" y="66" text-anchor="middle" font-weight="650">Trajectories</text>
      <text x="635" y="84" text-anchor="middle" font-size="11" fill="var(--dim)">
        every step recorded</text>

      <rect x="275" y="150" width="200" height="56" rx="8" fill="var(--accent-soft)"
            stroke="var(--accent)"/>
      <text x="375" y="174" text-anchor="middle" font-weight="650">Provider cache</text>
      <text x="375" y="192" text-anchor="middle" font-size="11" fill="var(--dim)">
        record once, replay forever</text>

      <rect x="535" y="150" width="200" height="56" rx="8" fill="var(--panel)"
            stroke="var(--line)"/>
      <text x="635" y="174" text-anchor="middle" font-weight="650">Metrics + judge</text>
      <text x="635" y="192" text-anchor="middle" font-size="11" fill="var(--dim)">
        ~42 metrics, per sample</text>

      <rect x="745" y="42" width="125" height="164" rx="8" fill="var(--panel)"
            stroke="var(--line)"/>
      <text x="807" y="112" text-anchor="middle" font-weight="650">Statistics</text>
      <text x="807" y="130" text-anchor="middle" font-size="11" fill="var(--dim)">paired</text>
      <text x="807" y="146" text-anchor="middle" font-size="11" fill="var(--dim)">clustered</text>
      <text x="807" y="162" text-anchor="middle" font-size="11" fill="var(--dim)">BH-adjusted</text>

      <text x="122" y="150" font-size="11" fill="var(--dim)">Hours, in the background</text>
      <text x="122" y="168" font-size="11" fill="var(--dim)">↓</text>
      <text x="122" y="186" font-size="11" fill="var(--accent)" font-weight="650">
        Seconds, in CI</text>
    </g>
  </svg>
  </div>
</section>

<section class="wrap">
  <p class="eyebrow">It keeps running</p>
  <h2>The evidence base grows, and its history is the git history</h2>
  <p>
    AgentGate is not a report. It records models against suites over time into one store, and
    tracks what it knows, how sure it is, and what is most worth measuring next. Every recording
    session commits a diff of what was learned.
  </p>
  <p>
    <strong>The leaderboard reports tiers, not positions.</strong> A model joins a tier while its
    interval overlaps the tier leader's; within a tier, the answer is <em>we cannot separate
    these</em>. Printing 1, 2, 3 by point estimate would assert an ordering between every adjacent
    pair at no stated confidence — the exact error the gate exists to prevent.
  </p>

  <h3 style="margin-top:2rem">Measured so far — __N_MODELS__ model(s) on real benchmark tasks</h3>
  <div class="scroll">
    <table>
      <thead><tr><th>Model</th><th>Metric</th><th class="num">Value</th>
        <th class="num">95% CI</th><th class="num">n</th></tr></thead>
      <tbody>__EVIDENCE_ROWS__</tbody>
    </table>
  </div>
  <p class="dim" style="font-size:.85rem;margin-top:.9rem">
    From τ²-bench retail (MIT), adapted to single-turn — <strong>not</strong> τ²-bench leaderboard
    scores. Note the pattern: fluent output, coherent phrasing, and the task still not done. That
    gap is the reason this project exists.
  </p>
</section>

<footer class="wrap">
  <p>
    Built by <a href="https://github.com/vedant1711">vedant1711</a> ·
    Apache-2.0 · generated __GENERATED__ ·
    <a href="docs/">docs</a> ·
    <a href="docs/limitations.md">limitations</a> ·
    <a href="https://github.com/vedant1711/agentgate">source</a>
  </p>
  <p style="font-size:.8rem">
    Read the limitations page before trusting any number here. The suites are small relative to
    production traffic, the judge is a model with its own measured biases, and every caveat that
    applies is stated rather than omitted.
  </p>
</footer>

<script>
const DATA = __DATA__;
const BLURB = __BLURB__;
const $ = id => document.getElementById(id);

let state = { scenario: Object.keys(DATA)[0], metric: 0, step: 0, alpha: 0.05 };

function panel() { return DATA[state.scenario].panels[state.metric]; }

function point() {
  const p = panel();
  const alphas = DATA[state.scenario].alphas;
  const ai = alphas.indexOf(state.alpha);
  return p.grid[state.step * alphas.length + ai];
}

function fmt(x, digits = 3) {
  if (x === null || x === undefined) return "—";
  return Math.abs(x) >= 1000 ? x.toFixed(0) : x.toFixed(digits);
}
function pval(x) { return x === null ? "—" : (x < 0.0001 ? "<0.0001" : x.toFixed(4)); }

function initScenarios() {
  const sel = $("scenario");
  sel.innerHTML = Object.keys(DATA).map(k =>
    `<option value="${k}">${k.replace(/_/g, " ")} — ${DATA[k].gate_verdict}</option>`).join("");
  sel.value = state.scenario;
  sel.onchange = () => { state.scenario = sel.value; state.metric = 0; state.step = 0;
    initMetrics(); render(); };
}

function initMetrics() {
  const sel = $("metric");
  sel.innerHTML = DATA[state.scenario].panels.map((p, i) =>
    `<option value="${i}">${p.metric}</option>`).join("");
  sel.value = state.metric;
  sel.onchange = () => { state.metric = +sel.value; render(); };
  $("margin").max = DATA[state.scenario].margin_steps - 1;
}

function initAlphas() {
  const box = $("alphas");
  box.innerHTML = "";
  for (const a of DATA[state.scenario].alphas) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = a.toFixed(2);
    b.onclick = () => { state.alpha = a; render(); };
    box.appendChild(b);
  }
}

function drawPlot(p, margin) {
  const svg = $("plot");
  const W = 620, H = 150, PADX = 26, MID = 88;
  const units = p.units;
  const extent = Math.max(margin * 1.15, ...units.map(Math.abs), 0.02);
  const x = v => PADX + (v + extent) / (2 * extent) * (W - 2 * PADX);
  const parts = [];

  // zero line and margin threshold
  parts.push(`<line x1="${x(0)}" y1="18" x2="${x(0)}" y2="${MID + 26}"
    stroke="var(--line)" stroke-width="1"/>`);
  parts.push(`<text x="${x(0)}" y="14" text-anchor="middle" font-size="10"
    fill="var(--dim)">no change</text>`);

  const worse = -margin;
  parts.push(`<line x1="${x(worse)}" y1="24" x2="${x(worse)}" y2="${MID + 22}"
    stroke="var(--regression)" stroke-width="1.6" stroke-dasharray="5 3"/>`);
  parts.push(`<text x="${x(worse)}" y="${MID + 38}" text-anchor="middle" font-size="10"
    fill="var(--regression)">margin −${fmt(margin)}</text>`);

  // shade the "worse than the margin" region
  parts.push(`<rect x="${PADX}" y="24" width="${Math.max(0, x(worse) - PADX)}"
    height="${MID - 2}" fill="var(--regression)" opacity=".07"/>`);

  // the actual per-cluster differences
  units.forEach((v, i) => {
    const jitter = ((i % 5) - 2) * 7;
    const colour = v < worse ? "var(--regression)"
      : (v < 0 ? "var(--underpowered)" : "var(--pass)");
    parts.push(`<circle cx="${x(v)}" cy="${MID / 2 + 14 + jitter}" r="5"
      fill="${colour}" opacity=".75"><title>cluster difference ${fmt(v)}</title></circle>`);
  });

  // the confidence interval for the mean difference
  if (p.ci_low !== null && p.ci_high !== null) {
    const y = MID + 8;
    parts.push(`<line x1="${x(p.ci_low)}" y1="${y}" x2="${x(p.ci_high)}" y2="${y}"
      stroke="var(--ink)" stroke-width="2"/>`);
    for (const b of [p.ci_low, p.ci_high]) {
      parts.push(`<line x1="${x(b)}" y1="${y - 5}" x2="${x(b)}" y2="${y + 5}"
        stroke="var(--ink)" stroke-width="2"/>`);
    }
    parts.push(`<circle cx="${x(p.delta)}" cy="${y}" r="4.5" fill="var(--ink)"/>`);
  }
  svg.innerHTML = parts.join("");
}

function render() {
  const p = panel();
  const alphas = DATA[state.scenario].alphas;
  [...$("alphas").children].forEach((b, i) =>
    b.setAttribute("aria-pressed", alphas[i] === state.alpha));

  $("margin").value = state.step;
  const pt = point();
  $("margin-value").textContent = fmt(pt.m);

  const chip = $("verdict");
  chip.textContent = pt.v;
  chip.className = "chip v-" + pt.v;
  $("verdict-note").innerHTML = BLURB[pt.v] || "";

  $("s-base").textContent = fmt(p.baseline);
  $("s-cand").textContent = fmt(p.candidate);
  $("s-delta").textContent = (p.delta >= 0 ? "+" : "") + fmt(p.delta);
  $("s-ci").textContent = p.ci_low === null ? "—" : `[${fmt(p.ci_low)}, ${fmt(p.ci_high)}]`;
  $("s-pr").textContent = pval(pt.pr);
  $("s-pn").textContent = pval(pt.pn);
  $("s-mde").textContent = fmt(pt.mde);
  $("s-n").textContent = p.n_units + (p.clustered ? " clusters" : " tasks");

  $("plot-caption").textContent =
    `Each dot is one ${p.clustered ? "cluster" : "task"}: candidate minus baseline on ` +
    `${p.metric}. The bar is the 95% interval for the mean difference. Anything left of the ` +
    `dashed line is worse than the margin.`;

  drawPlot(p, pt.m);
}

initScenarios(); initMetrics(); initAlphas();
$("margin").oninput = e => { state.step = +e.target.value; render(); };
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    build()
