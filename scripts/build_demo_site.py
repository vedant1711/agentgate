"""Build the landing page: an evaluation console, not an article.

The page has one job — make a reader who evaluates AI systems for a living believe, within about
fifteen seconds, that the person who built this knows what they are doing. Prose cannot do that.
A reader like that is convinced by **the artefacts the field actually produces**: a forest plot of
effect sizes with confidence intervals, a named margin, a BH-adjusted p-value, an explicit
minimum detectable effect.

So the centrepiece is a working gate console showing exactly those, computed by the real engine.
Everything else on the page supports it: what the change was, what a naive check would have
concluded, which techniques produced the numbers, and what two real models scored.

Chart decisions, made deliberately rather than by default:

* **Forest plot** for per-metric effects, because the question is not "how big is each number" but
  "which interval crosses the line" — the reader's eye performs the test.
* **Proportion metrics only on that axis.** Token counts move in the hundreds and success rates in
  hundredths; one axis for both would need a second scale, and a dual-scale chart is the fastest
  way to make a rigorous plot lie. Efficiency metrics get their own table in their own units.
* **Status colour plus an always-present label**, never colour alone.
* **Dot plot with whiskers** for the model comparison, because intervals are the point; bars would
  imply the point estimate is the finding.

Usage::

    uv run python scripts/build_demo_site.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentgate.demo import run_scenario
from agentgate.report.chrome import REPO_URL
from agentgate.report.playground import panel_payload, panels_for
from agentgate.report.stack import STACK
from agentgate.report.story import (
    copy_for,
    efficiency_rows,
    forest_rows,
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

# Ordered as an argument, not alphabetically: the two cases where a threshold check is wrong come
# first, one in each direction, because those are the whole point. The rest show the gate agreeing
# with common sense, which is what makes the first two credible rather than contrarian.
SCENARIOS = ("verbosity_attack", "injection", "dropped_tool", "flaky_dependency", "no_op")
STOP_VERDICTS = frozenset({"REGRESSION", "SAFETY_FAIL"})


def build_scenario(key: str) -> dict[str, Any]:
    """Run one scenario through the real pipeline and assemble everything the page renders."""
    result = run_scenario(key, suite_path=SUITE)
    baseline = [s for s in result.baseline_scores if s.metric == HEADLINE_METRIC]
    candidate = [s for s in result.candidate_scores if s.metric == HEADLINE_METRIC]

    outcomes = task_outcomes(baseline, candidate)
    naive = naive_verdict(outcomes)
    copy = copy_for(key)
    verdict = verdict_copy(result.verdict)
    rulings = result.gate.rulings
    panels = panels_for(result.gate, limit=4)
    payload = panel_payload(panels, scenario=key, suite="crm_ops")

    blocking = [r for r in rulings if r.blocks]
    return {
        "key": key,
        "title": copy.title,
        "change": copy.change,
        "why": copy.why_it_matters,
        "lesson": copy.lesson,
        "verdict": result.verdict,
        "headline": verdict["headline"],
        "body": verdict["body"],
        "agrees": (result.verdict in STOP_VERDICTS) == naive.would_block,
        "n_metrics": len(rulings),
        "n_blocking": len(blocking),
        "naive": {
            "before": round(naive.baseline_rate, 4),
            "after": round(naive.candidate_rate, 4),
            "threshold": naive.threshold,
            "verdict": naive.verdict,
            "describe": naive.describe(),
        },
        "forest": forest_rows(rulings),
        "efficiency": efficiency_rows(rulings),
        "tasks": outcome_payload(outcomes),
        "panels": payload["panels"],
        "alphas": payload["alphas"],
        "labels": {panel.metric: plain_label(panel.metric) for panel in panels},
    }


def collect() -> dict[str, Any]:
    """Run every scenario shown on the page."""
    print("running scenarios through the real pipeline:")
    scenarios: dict[str, Any] = {}
    for key in SCENARIOS:
        item = build_scenario(key)
        scenarios[key] = item
        flag = "" if item["agrees"] else "   <- a threshold check is wrong here"
        print(
            f"  {key:<18} gate={item['verdict']:<13} naive={item['naive']['verdict']:<5}"
            f" forest={len(item['forest'])}{flag}"
        )
    return {"order": list(SCENARIOS), "scenarios": scenarios}


def evidence() -> dict[str, Any]:
    """Load the committed evidence snapshot from the real-model recordings."""
    if not RESULTS.exists():
        return {"cells": []}
    loaded: dict[str, Any] = json.loads(RESULTS.read_text(encoding="utf-8"))
    return loaded


COMPARE_METRICS = (
    ("outcome.task_success", "Finished the job"),
    ("judge.instruction_following", "Followed instructions"),
    ("trajectory.recall", "Found the right tools"),
    ("trajectory.argument_correctness", "Got arguments right"),
    ("judge.coherence", "Sounded coherent"),
)


def model_payload(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Shape the recorded-model evidence for the comparison chart."""
    cells = [c for c in snapshot.get("cells", []) if c.get("complete")]
    models = [{"label": c["label"], "id": c["model_id"]} for c in cells]
    rows = []
    for metric, label in COMPARE_METRICS:
        series = []
        for cell in cells:
            found = next((m for m in cell["metrics"] if m["metric"] == metric), None)
            series.append(
                None
                if found is None
                else {
                    "v": round(float(found["value"]), 4),
                    "lo": None if found.get("ci_low") is None else round(found["ci_low"], 4),
                    "hi": None if found.get("ci_high") is None else round(found["ci_high"], 4),
                }
            )
        if any(series):
            rows.append({"label": label, "series": series})
    tokens = []
    for cell in cells:
        found = next((m for m in cell["metrics"] if m["metric"] == "efficiency.total_tokens"), None)
        tokens.append(None if found is None else round(float(found["value"])))
    return {"models": models, "rows": rows, "tokens": tokens}


def stack_payload() -> list[dict[str, Any]]:
    """Serialise the technique showcase."""
    return [
        {
            "title": group.title,
            "blurb": group.blurb,
            "items": [asdict(item) for item in group.items],
        }
        for group in STACK
    ]


def build() -> Path:
    """Generate ``site/index.html``."""
    snapshot = evidence()
    data = {
        **collect(),
        "models": model_payload(snapshot),
        "stack": stack_payload(),
        "generated": datetime.now(UTC).strftime("%B %Y"),
        "repo": REPO_URL,
    }
    SITE.mkdir(parents=True, exist_ok=True)
    target = SITE / "index.html"
    target.write_text(
        PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__REPO__", REPO_URL)
        .replace("__GENERATED__", data["generated"]),
        encoding="utf-8",
    )
    print(f"wrote {target} ({target.stat().st_size / 1024:.0f} KB)")
    return target


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

STYLE = """
:root{
  --bg:#f7f7f4; --surface:#ffffff; --sunken:#f1f0ec; --line:#e2e0d8; --line-soft:#eeece5;
  --ink:#14171c; --ink-2:#4c545f; --ink-3:#7c848f;
  --brand:#0f6e64; --brand-soft:#e2f0ed; --on-brand:#ffffff;
  --s1:#2a78d6; --s2:#eb6834;
  --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --critical:#d03b3b;
  --good-bg:#e7f5e7; --warn-bg:#fdf1d8; --crit-bg:#fae6e6; --stop-bg:#efe7fa; --stop:#6b3fb5;
  --grid:#e9e7e0; --axis:#c9c6bc;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  color-scheme:light;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0c0e11; --surface:#14181d; --sunken:#101317; --line:#252b33; --line-soft:#1c2128;
  --ink:#e8ecf1; --ink-2:#a6b0bd; --ink-3:#78828f;
  --brand:#4fd6c4; --brand-soft:#0e2f2b; --on-brand:#0c0e11;
  --s1:#3987e5; --s2:#d95926;
  --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --critical:#d03b3b;
  --good-bg:#0f2a12; --warn-bg:#2e2410; --crit-bg:#2e1414; --stop-bg:#201936; --stop:#bda6f5;
  --grid:#1e242b; --axis:#333b45;
  color-scheme:dark;
}}
:root[data-theme="dark"]{
  --bg:#0c0e11; --surface:#14181d; --sunken:#101317; --line:#252b33; --line-soft:#1c2128;
  --ink:#e8ecf1; --ink-2:#a6b0bd; --ink-3:#78828f;
  --brand:#4fd6c4; --brand-soft:#0e2f2b; --on-brand:#0c0e11;
  --s1:#3987e5; --s2:#d95926;
  --good:#0ca30c; --warn:#fab219; --serious:#ec835a; --critical:#d03b3b;
  --good-bg:#0f2a12; --warn-bg:#2e2410; --crit-bg:#2e1414; --stop-bg:#201936; --stop:#bda6f5;
  --grid:#1e242b; --axis:#333b45;
  color-scheme:dark;
}

*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;scroll-behavior:smooth;scroll-padding-top:64px}
body{margin:0;background:var(--bg);color:var(--ink);font:15.5px/1.6 ui-sans-serif,-apple-system,
  BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;font-variant-numeric:tabular-nums}
.wrap{max-width:1120px;margin:0 auto;padding:0 1.5rem}
h1,h2,h3,h4{margin:0;letter-spacing:-.02em;text-wrap:balance;font-weight:640}
h1{font-size:clamp(2.1rem,4.6vw,3.4rem);line-height:1.05;letter-spacing:-.035em}
h2{font-size:clamp(1.5rem,2.6vw,2rem);line-height:1.15}
h3{font-size:1.02rem}
p{margin:0 0 1rem;max-width:68ch;color:var(--ink-2)}
a{color:var(--brand);text-underline-offset:.18em}
.mono{font-family:var(--mono)}
.eyebrow{font:600 .68rem/1 var(--mono);letter-spacing:.16em;text-transform:uppercase;
  color:var(--brand);margin:0 0 1rem}
.dim{color:var(--ink-3)}

/* nav */
.topbar{position:sticky;top:0;z-index:50;background:color-mix(in srgb,var(--bg) 88%,transparent);
  backdrop-filter:saturate(180%) blur(12px);border-bottom:1px solid var(--line)}
.topbar .inner{max-width:1120px;margin:0 auto;padding:.7rem 1.5rem;display:flex;
  align-items:center;gap:2rem}
.brand{display:flex;align-items:center;gap:.5rem;font-weight:680;font-size:.98rem;
  text-decoration:none;color:var(--ink);letter-spacing:-.02em}
.brand .dot{width:9px;height:9px;border-radius:2px;background:var(--brand)}
.topbar nav{display:flex;gap:1.4rem;margin-left:auto;align-items:center}
.topbar nav a{font-size:.85rem;text-decoration:none;color:var(--ink-2)}
.topbar nav a:hover{color:var(--ink)}
.ghost{border:1px solid var(--line);border-radius:6px;padding:.32rem .7rem !important;
  color:var(--ink) !important}
@media(max-width:820px){.topbar nav a.opt{display:none}}

/* hero */
.hero{padding:5rem 0 3rem}
.hero .lede{font-size:clamp(1.05rem,1.6vw,1.22rem);max-width:60ch;color:var(--ink-2);
  margin-top:1.2rem}
.cta{display:flex;gap:.7rem;flex-wrap:wrap;margin-top:2rem}
.btn{display:inline-flex;align-items:center;gap:.4rem;padding:.62rem 1.1rem;border-radius:7px;
  text-decoration:none;font-size:.9rem;font-weight:560;border:1px solid var(--line);
  color:var(--ink);background:var(--surface)}
.btn.primary{background:var(--brand);border-color:var(--brand);color:var(--on-brand)}
.btn:hover{border-color:var(--brand)}

.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  border:1px solid var(--line);border-radius:12px;overflow:hidden;margin-top:2.8rem;
  background:var(--surface)}
.stat{padding:1.1rem 1.2rem;border-right:1px solid var(--line-soft)}
.stat:last-child{border-right:0}
.stat .n{font:660 1.55rem/1 var(--mono);letter-spacing:-.03em;display:block}
.stat .k{font-size:.74rem;color:var(--ink-3);margin-top:.35rem;display:block;line-height:1.35}

section{padding:4rem 0;border-top:1px solid var(--line)}
.shead{margin-bottom:1.8rem}

/* console */
.console{border:1px solid var(--line);border-radius:14px;background:var(--surface);
  overflow:hidden}
.console .bar{display:flex;align-items:center;gap:.6rem;padding:.7rem 1rem;
  border-bottom:1px solid var(--line);background:var(--sunken);font:500 .8rem/1 var(--mono);
  color:var(--ink-3);flex-wrap:wrap}
.console .bar .dots{display:flex;gap:.32rem;margin-right:.4rem}
.console .bar .dots i{width:9px;height:9px;border-radius:50%;background:var(--axis)}
.tabs{display:flex;gap:.3rem;padding:.7rem 1rem 0;flex-wrap:wrap;
  border-bottom:1px solid var(--line)}
.tabs button{border:1px solid transparent;border-bottom:none;background:none;cursor:pointer;
  font:inherit;font-size:.82rem;color:var(--ink-3);padding:.5rem .8rem;border-radius:7px 7px 0 0}
.tabs button[aria-selected=true]{background:var(--surface);border-color:var(--line);
  color:var(--ink);font-weight:600;margin-bottom:-1px}
.tabs button:hover{color:var(--ink)}

.verdict-head{display:flex;gap:1.2rem;align-items:flex-start;padding:1.4rem 1.4rem 1.1rem;
  flex-wrap:wrap}
.vbadge{display:inline-flex;align-items:center;gap:.45rem;padding:.42rem .8rem;border-radius:7px;
  font:700 .76rem/1 var(--mono);letter-spacing:.06em;white-space:nowrap}
.vbadge.good{background:var(--good-bg);color:var(--good)}
.vbadge.warn{background:var(--warn-bg);color:#8a5f04}
.vbadge.crit{background:var(--crit-bg);color:var(--critical)}
.vbadge.stop{background:var(--stop-bg);color:var(--stop)}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]) .vbadge.warn{color:var(--warn)}}
:root[data-theme=dark] .vbadge.warn{color:var(--warn)}

.chip{display:inline-flex;align-items:center;gap:.35rem;padding:.16rem .5rem;border-radius:5px;
  font:600 .68rem/1.5 var(--mono);letter-spacing:.03em}
.chip.good{background:var(--good-bg);color:var(--good)}
.chip.warn{background:var(--warn-bg);color:#8a5f04}
.chip.crit{background:var(--crit-bg);color:var(--critical)}
.chip.stop{background:var(--stop-bg);color:var(--stop)}
.chip.flat{background:var(--sunken);color:var(--ink-3)}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]) .chip.warn{color:var(--warn)}}
:root[data-theme=dark] .chip.warn{color:var(--warn)}

.split{display:grid;grid-template-columns:minmax(0,1fr) 264px;gap:0;
  border-top:1px solid var(--line)}
@media(max-width:900px){.split{grid-template-columns:1fr}}
.plot{padding:1.2rem 1.4rem 1.4rem;min-width:0}
.side{border-left:1px solid var(--line);padding:1.2rem 1.3rem;background:var(--sunken)}
@media(max-width:900px){.side{border-left:0;border-top:1px solid var(--line)}}
.side h4{font:600 .68rem/1 var(--mono);letter-spacing:.13em;text-transform:uppercase;
  color:var(--ink-3);margin-bottom:.9rem}
.kv{display:flex;justify-content:space-between;gap:.8rem;padding:.42rem 0;
  border-bottom:1px dotted var(--line);font-size:.82rem}
.kv:last-of-type{border-bottom:0}
.kv span{color:var(--ink-3)}
.kv b{font:600 .82rem var(--mono)}

.plot-title{font-size:.86rem;font-weight:620;margin-bottom:.15rem}
.plot-sub{font-size:.78rem;color:var(--ink-3);margin-bottom:.9rem}
svg.forest{width:100%;height:auto;display:block;overflow:visible}
.legend{display:flex;gap:1rem;flex-wrap:wrap;font-size:.74rem;color:var(--ink-3);
  margin-top:.9rem;padding-top:.7rem;border-top:1px solid var(--line-soft)}
.legend i{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:.32rem;
  vertical-align:-1px}

.compare{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line);
  border-top:1px solid var(--line)}
@media(max-width:700px){.compare{grid-template-columns:1fr}}
.compare > div{background:var(--surface);padding:1.1rem 1.4rem}
.compare h4{font:600 .68rem/1 var(--mono);letter-spacing:.13em;text-transform:uppercase;
  color:var(--ink-3);margin-bottom:.7rem}
.compare p{font-size:.85rem;margin:.6rem 0 0}
.note{padding:1rem 1.4rem;font-size:.86rem;border-top:1px solid var(--line)}
.note.bad{background:var(--warn-bg);color:var(--ink)}

/* generic */
.grid3{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(250px,1fr))}
.card{border:1px solid var(--line);border-radius:11px;padding:1.2rem;background:var(--surface)}
.card p{font-size:.87rem;margin:.4rem 0 0}

/* stack */
.stackgrid{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.sgroup{border:1px solid var(--line);border-radius:11px;background:var(--surface);
  padding:1.2rem;display:flex;flex-direction:column}
.sgroup > h3{margin-bottom:.3rem}
.sgroup > p{font-size:.83rem;margin-bottom:1rem}
.tech{border-top:1px solid var(--line-soft);padding:.7rem 0}
.tech:last-child{padding-bottom:0}
.tech .t{display:flex;align-items:baseline;gap:.5rem;flex-wrap:wrap}
.tech .t b{font-size:.88rem;font-weight:620;color:var(--ink)}
.tech .t code{font:500 .68rem var(--mono);color:var(--ink-3);background:var(--sunken);
  padding:.1rem .35rem;border-radius:4px}
.tech .d{font-size:.82rem;color:var(--ink-2);margin-top:.2rem}

/* model chart */
.mchart{border:1px solid var(--line);border-radius:12px;background:var(--surface);padding:1.4rem}
.mlegend{display:flex;gap:1.2rem;flex-wrap:wrap;font-size:.82rem;margin-bottom:1.2rem}
.mlegend span{display:inline-flex;align-items:center;gap:.4rem}
.mlegend i{width:10px;height:10px;border-radius:50%;display:inline-block}

table{width:100%;border-collapse:collapse;font-size:.85rem}
th,td{text-align:left;padding:.5rem .6rem;border-bottom:1px solid var(--line-soft)}
th{font:600 .68rem var(--mono);letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3)}
td.n,th.n{text-align:right;font-family:var(--mono)}
.scroll{overflow-x:auto}

pre{background:var(--sunken);border:1px solid var(--line);border-radius:9px;padding:1rem;
  overflow-x:auto;font:.8rem/1.6 var(--mono);margin:0}
pre .c{color:var(--ink-3)}
pre .k{color:var(--brand);font-weight:600}

footer{padding:3rem 0 4rem;border-top:1px solid var(--line);font-size:.84rem;color:var(--ink-3)}
footer a{color:var(--ink-2)}
"""

BODY = """
<div class="topbar"><div class="inner">
  <a class="brand" href="./"><span class="dot"></span>AgentGate</a>
  <nav>
    <a href="#gate">Live gate</a>
    <a href="#pipeline" class="opt">Pipeline</a>
    <a href="#stack">Eval stack</a>
    <a href="#models" class="opt">Results</a>
    <a href="docs/">Docs</a>
    <a href="gallery.html" class="opt">Reports</a>
    <a class="ghost" href="__REPO__">GitHub</a>
  </nav>
</div></div>

<header class="hero wrap">
  <p class="eyebrow">Statistical evaluation infrastructure for LLM agents</p>
  <h1>Ship agent changes on<br>evidence, not vibes.</h1>
  <p class="lede">
    A CI gate that blocks a pull request only when an agent has <em>statistically
    significantly</em> regressed — with paired non-inferiority tests, cluster-robust errors,
    FDR control across 42 metrics, and an <span class="mono">UNDERPOWERED</span> verdict for when
    the evidence genuinely cannot tell.
  </p>
  <div class="cta">
    <a class="btn primary" href="#gate">Open the gate console</a>
    <a class="btn" href="docs/architecture/">Architecture</a>
    <a class="btn" href="__REPO__">Source</a>
  </div>
  <div class="stats" id="stats"></div>
</header>

<section id="gate" class="wrap">
  <div class="shead">
    <p class="eyebrow">Live gate console</p>
    <h2>Every number below was computed by the real engine</h2>
    <p>
      Pick a change a developer might have made. The suite is 70 tasks in 14 clusters, both sides
      run on identical task instances with identical seeds. Nothing here is illustrative.
    </p>
  </div>

  <div class="console">
    <div class="bar">
      <span class="dots"><i></i><i></i><i></i></span>
      <span id="cmdline">agentgate compare --baseline main --candidate pr-482</span>
    </div>
    <div class="tabs" id="tabs" role="tablist"></div>

    <div class="verdict-head">
      <div id="vbadge"></div>
      <div style="flex:1;min-width:260px">
        <h3 id="vtitle" style="margin-bottom:.25rem"></h3>
        <p id="vbody" style="font-size:.88rem;margin:0"></p>
      </div>
    </div>

    <div class="split">
      <div class="plot">
        <div class="plot-title">Effect on each gated metric</div>
        <div class="plot-sub" id="plot-sub"></div>
        <svg class="forest" id="forest" role="img" aria-labelledby="forest-desc"></svg>
        <p id="forest-desc" class="visually-hidden" style="position:absolute;left:-9999px"></p>
        <div class="legend">
          <span><i style="background:var(--critical)"></i>Blocks the merge</span>
          <span><i style="background:var(--warn)"></i>Not enough evidence</span>
          <span><i style="background:var(--good)"></i>No meaningful regression</span>
          <span>Shaded band = declared tolerance · dashed line = no change</span>
        </div>
      </div>
      <div class="side">
        <h4>Run</h4>
        <div class="kv"><span>Paired tasks</span><b id="k-tasks">—</b></div>
        <div class="kv"><span>Independent units</span><b id="k-clusters">—</b></div>
        <div class="kv"><span>Metrics gated</span><b id="k-metrics">—</b></div>
        <div class="kv"><span>Blocking</span><b id="k-block">—</b></div>
        <div class="kv"><span>Correction</span><b>BH-FDR</b></div>
        <div class="kv"><span>Alpha</span><b>0.05</b></div>
        <p class="dim" style="font-size:.72rem;margin:.7rem 0 0;line-height:1.45">
          Tests run on per-cluster means, so the independent sample size is the cluster count —
          not the task count. Using the larger number would overstate the power fivefold.
        </p>
        <h4 style="margin-top:1.4rem">Cost</h4>
        <div id="eff"></div>
      </div>
    </div>

    <div class="compare">
      <div>
        <h4>A conventional CI check</h4>
        <div id="naive-badge"></div>
        <p id="naive-why"></p>
      </div>
      <div>
        <h4>AgentGate</h4>
        <div id="gate-badge"></div>
        <p id="gate-why"></p>
      </div>
    </div>
    <div id="note"></div>
  </div>

  <p class="dim" style="font-size:.82rem;margin-top:1rem">
    The forest plot shows proportion-scale metrics only. Token and latency deltas live in the Cost
    panel because they are measured in different units — putting them on one axis would need a
    second scale, and a dual-scale chart is the fastest way to make a rigorous plot mislead.
  </p>
</section>

<section id="pipeline" class="wrap">
  <div class="shead">
    <p class="eyebrow">Pipeline</p>
    <h2>Seven layers, two speeds</h2>
    <p>
      Recording a suite against a local model takes hours. Deciding, once the responses exist,
      takes seconds. The provider layer records once and replays forever, so CI never calls a
      model and a verdict reproduces byte for byte.
    </p>
  </div>
  <div id="pipeline-svg"></div>
  <div class="grid3" style="margin-top:1.5rem">
    <div class="card">
      <h3>Trajectory is the hinge</h3>
      <p>Everything above it produces one; everything below only reads one. Metrics, statistics
      and the gate are all testable against recorded trajectories — no model, no network.</p>
    </div>
    <div class="card">
      <h3>Replay fails loudly</h3>
      <p>A cache miss in CI is an error, never a silent live call. That single decision is what
      makes the verdict deterministic and the pipeline free to run.</p>
    </div>
    <div class="card">
      <h3>Per-sample, never aggregated</h3>
      <p>Scores are stored per (task, repetition) in DuckDB, so any run can be re-analysed at a
      different K, margin or alpha without re-running the agent.</p>
    </div>
  </div>
</section>

<section id="stack" class="wrap">
  <div class="shead">
    <p class="eyebrow">Evaluation stack</p>
    <h2>26 techniques, each solving a named failure</h2>
    <p>
      Every entry names the module that implements it. A test imports all of them, so nothing on
      this page can claim a capability the codebase does not have.
    </p>
  </div>
  <div class="stackgrid" id="stack"></div>
</section>

<section id="models" class="wrap">
  <div class="shead">
    <p class="eyebrow">Recorded evidence</p>
    <h2>Two open models, 111 published benchmark tasks</h2>
    <p>
      666 runs and 4.5&nbsp;million tokens of real inference against τ²-bench retail, graded
      against its own gold trajectories. Whiskers are 95% cluster-robust intervals over 111
      independent clusters.
    </p>
  </div>
  <div class="mchart">
    <div class="mlegend" id="mlegend"></div>
    <svg id="models-svg" role="img"
      style="width:100%;height:auto;display:block;overflow:visible"></svg>
    <div class="scroll" style="margin-top:1.5rem">
      <table id="mtable"></table>
    </div>
  </div>
  <div class="grid3" style="margin-top:1.5rem">
    <div class="card">
      <h3>Coherent and wrong</h3>
      <p>The larger model sounds <em>more</em> coherent while finishing fewer jobs. That gap is
      why "it reads well" is not a measurement.</p>
    </div>
    <div class="card">
      <h3>The gate refuses to rank them</h3>
      <p>0.180 vs 0.111 looks like a 62% relative difference. Paired across 111 tasks it is
      −0.069 [−0.159, 0.021], and the suite's minimum detectable effect is 0.115.</p>
    </div>
    <div class="card">
      <h3>Safety is not averaged</h3>
      <p>Unconfirmed destructive actions fired on 27 of 111 tasks for the larger model, and never
      for the smaller. That bypasses the statistics entirely.</p>
    </div>
  </div>
</section>

<section class="wrap">
  <div class="shead">
    <p class="eyebrow">Interface</p>
    <h2>It is a CLI and a library, not a notebook</h2>
  </div>
  <div class="grid3" style="grid-template-columns:minmax(0,1.15fr) minmax(0,1fr)">
    <pre><span class="c"># record once — slow, in the background</span>
$ agentgate run --suite suites/tau2_retail \\
    --model ollama_chat/llama3.2:3b --mode cache

<span class="c"># decide in CI — instant, offline, free</span>
$ agentgate compare --baseline main --candidate pr-482
<span class="k">REGRESSION</span>  trajectory.in_order_match
           -0.243 [-0.371, -0.115]  p_adj=0.0004
exit 2

<span class="c"># what has the harness learned so far?</span>
$ agentgate leaderboard --suite tau2_retail
tier  model          task_success   95% CI
1     Llama 3.2 3B   0.180          [0.108, 0.252]
1     Qwen2.5 7B     0.111          [0.053, 0.169]
<span class="c"># tier 1 twice: not separable on this evidence</span></pre>
    <div style="display:flex;flex-direction:column;gap:1rem">
      <div class="card">
        <h3>Exit code is the verdict</h3>
        <p>0 pass, 2 regression, 3 underpowered, 4 safety. Drop it into any CI system without
        parsing anything.</p>
      </div>
      <div class="card">
        <h3>Every artefact is generated</h3>
        <p>This page, the metric catalogue, the results tables and the PR comment all come from
        the pipeline. A drift check fails the build if a committed page stops matching its data.</p>
      </div>
    </div>
  </div>
</section>

<section class="wrap">
  <div class="shead">
    <p class="eyebrow">Engineering</p>
    <h2>Built to be trusted, not just to run</h2>
  </div>
  <div class="grid3">
    <div class="card"><h3>779 tests</h3><p>Unit, property-based with Hypothesis, end-to-end, and
      simulation tests that verify the gate's false-positive rate against synthetic ground
      truth.</p></div>
    <div class="card"><h3><span class="mono">mypy --strict</span></h3><p>134 source files, zero
      escapes. Every boundary is a pydantic v2 model exported as JSON Schema.</p></div>
    <div class="card"><h3>Golden values</h3><p>Every metric is pinned to a hand-computed expected
      value, so a refactor cannot silently change what a number means.</p></div>
    <div class="card"><h3>Reproducible by construction</h3><p>Suite hash, seeds, prompt hashes,
      model pins and library versions all fold into one config hash. Host details are excluded on
      purpose.</p></div>
    <div class="card"><h3>Zero paid resources</h3><p>Runs entirely on a laptop with local
      models or free tiers. Cloning it costs nothing — a hard constraint, not a demo
      mode.</p></div>
    <div class="card"><h3>Self-recording</h3><p>A weekly workflow records new evidence, commits
      the snapshot, and redeploys — so the git history is the history of what it has
      learned.</p></div>
  </div>
</section>

<footer class="wrap">
  <p>
    <strong style="color:var(--ink-2)">AgentGate</strong> ·
    <a href="__REPO__">github.com/vedant1711/agentgate</a> ·
    Apache-2.0 · generated __GENERATED__
  </p>
  <p style="max-width:62ch;font-size:.8rem">
    Read the <a href="docs/limitations/">limitations</a> before trusting any number here. The
    suites are small next to production traffic, the judge is itself a model with measured biases,
    and the τ² adaptation is single-turn where the original is multi-turn. All of it is stated
    rather than omitted — the same discipline as the UNDERPOWERED verdict, applied to the project.
  </p>
</footer>
"""

SCRIPT = r"""
const D = __DATA__;
const $ = id => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
let cur = D.order[0];

const VCLASS = {PASS:"good", REGRESSION:"crit", UNDERPOWERED:"warn", SAFETY_FAIL:"stop",
                INCONCLUSIVE:"flat"};
const VCOLOR = {PASS:"var(--good)", REGRESSION:"var(--critical)", UNDERPOWERED:"var(--warn)",
                SAFETY_FAIL:"var(--stop)", INCONCLUSIVE:"var(--ink-3)"};
const VSHORT = {PASS:"no regression", REGRESSION:"blocks", UNDERPOWERED:"insufficient",
                SAFETY_FAIL:"safety", INCONCLUSIVE:"n/a"};
const pct = x => (x*100).toFixed(0) + "%";

function el(tag, attrs, text){
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (text !== undefined) n.textContent = text;
  return n;
}

/* ---------------- hero stats ---------------- */
function renderStats(){
  const items = [
    ["42", "metrics scored per sample"],
    ["111", "τ²-bench tasks, 111 clusters"],
    ["666", "recorded agent runs"],
    ["4.5M", "tokens of real inference"],
    ["779", "tests, mypy --strict"],
  ];
  $("stats").innerHTML = items.map(([n,k]) =>
    `<div class="stat"><span class="n">${n}</span><span class="k">${k}</span></div>`).join("");
}

/* ---------------- tabs ---------------- */
function renderTabs(){
  $("tabs").innerHTML = D.order.map(k =>
    `<button role="tab" data-k="${k}">${D.scenarios[k].title}</button>`).join("");
  for (const b of $("tabs").children)
    b.onclick = () => { cur = b.dataset.k; render(); };
}

/* ---------------- forest plot ---------------- */
function forest(rows, margin){
  const svg = $("forest");
  svg.innerHTML = "";
  if (!rows.length){
    svg.appendChild(el("text", {x:0, y:20, fill:"var(--ink-3)", "font-size":"13"},
      "No proportion-scale metric was gated for this scenario."));
    svg.setAttribute("viewBox", "0 0 600 40");
    return;
  }
  const W = 640, padL = 190, padR = 96, rowH = 30, top = 26;
  const H = top + rows.length*rowH + 26;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

  let lo = Math.min(0, ...rows.map(r => r.lo)), hi = Math.max(0, ...rows.map(r => r.hi));
  const pad = Math.max(0.04, (hi-lo)*0.12); lo -= pad; hi += pad;
  const x = v => padL + ((v-lo)/(hi-lo))*(W-padL-padR);

  // tolerance band: from -margin to 0 is "acceptable worse"
  if (margin > 0){
    svg.appendChild(el("rect", {x:x(-margin), y:top-8, width:Math.max(1,x(0)-x(-margin)),
      height:rows.length*rowH+8, fill:"var(--brand)", opacity:"0.07"}));
  }
  // axis ticks
  const ticks = [lo, (lo+hi)/2, hi].map(v => Math.round(v*100)/100);
  for (const t of [...new Set(ticks)]){
    svg.appendChild(el("line", {x1:x(t), y1:top-8, x2:x(t), y2:top+rows.length*rowH,
      stroke:"var(--grid)", "stroke-width":"1"}));
    svg.appendChild(el("text", {x:x(t), y:H-8, fill:"var(--ink-3)", "font-size":"10.5",
      "text-anchor":"middle", "font-family":"var(--mono)"}, t.toFixed(2)));
  }
  // zero reference
  svg.appendChild(el("line", {x1:x(0), y1:top-10, x2:x(0), y2:top+rows.length*rowH+2,
    stroke:"var(--axis)", "stroke-width":"1.5", "stroke-dasharray":"3 3"}));

  rows.forEach((r, i) => {
    const y = top + i*rowH + rowH/2;
    const c = VCOLOR[r.verdict] || "var(--ink-3)";
    svg.appendChild(el("text", {x:padL-12, y:y+4, fill:"var(--ink-2)", "font-size":"12",
      "text-anchor":"end"}, r.label.length>28 ? r.label.slice(0,27)+"…" : r.label));
    // interval
    svg.appendChild(el("line", {x1:x(r.lo), y1:y, x2:x(r.hi), y2:y, stroke:c,
      "stroke-width":"2", "stroke-linecap":"round", opacity:"0.55"}));
    for (const v of [r.lo, r.hi])
      svg.appendChild(el("line", {x1:x(v), y1:y-4, x2:x(v), y2:y+4, stroke:c, "stroke-width":"2"}));
    // point estimate, ringed so it reads against the interval
    svg.appendChild(el("circle", {cx:x(r.delta), cy:y, r:"5", fill:c,
      stroke:"var(--surface)", "stroke-width":"2"}));
    // right-hand label: colour is never the only channel
    svg.appendChild(el("text", {x:W-padR+10, y:y+4, fill:"var(--ink-3)", "font-size":"10.5",
      "font-family":"var(--mono)"}, VSHORT[r.verdict] || ""));
    const ttl = el("title", {});
    ttl.textContent = `${r.label}: ${r.delta>=0?"+":""}${r.delta.toFixed(3)} `
      + `[${r.lo.toFixed(3)}, ${r.hi.toFixed(3)}]`
      + (r.p===null?"":` · BH-adjusted p=${r.p}`)
      + ` · n=${r.n_units} ${r.clustered ? "clusters" : "tasks"}`
      + ` from ${r.n_tasks} paired tasks`;
    svg.appendChild(ttl);
  });
  $("forest-desc").textContent = rows.map(r =>
    `${r.label} changed by ${r.delta.toFixed(3)} `
    + `(95% CI ${r.lo.toFixed(3)} to ${r.hi.toFixed(3)}), verdict ${r.verdict}.`).join(" ");
}

/* ---------------- main render ---------------- */
function render(){
  const s = D.scenarios[cur];
  for (const b of $("tabs").children)
    b.setAttribute("aria-selected", b.dataset.k === cur);

  $("cmdline").textContent =
    `agentgate compare --baseline main --candidate ${cur.replace(/_/g,"-")}`;

  const vc = VCLASS[s.verdict] || "flat";
  $("vbadge").innerHTML = `<span class="vbadge ${vc}">${s.verdict}</span>`;
  $("vtitle").textContent = s.headline;
  $("vbody").textContent = s.body;

  const margin = s.forest.length ? s.forest[0].margin : 0;
  const hidden = s.n_metrics - s.forest.length;
  $("plot-sub").textContent =
    `Change from baseline with 95% ${s.forest.length && s.forest[0].clustered
      ? "cluster-robust " : ""}confidence intervals. An interval clear of the shaded tolerance `
    + `band is a real effect.`
    + (hidden > 0 ? ` ${hidden} further metric${hidden>1?"s are":" is"} measured in other units `
        + `and shown under Cost.` : "");
  forest(s.forest, margin);

  $("k-tasks").textContent = s.tasks.length;
  $("k-clusters").textContent = s.forest.length ? s.forest[0].n_units : "—";
  $("k-metrics").textContent = s.n_metrics;
  $("k-block").textContent = s.n_blocking;

  $("eff").innerHTML = s.efficiency.length
    ? s.efficiency.map(e => {
        const d = e.delta >= 0 ? `+${e.delta}` : `${e.delta}`;
        return `<div class="kv"><span>${e.label}</span><b>${d}</b></div>`;
      }).join("")
    : `<div class="kv"><span>No cost change measured</span><b>—</b></div>`;

  const nb = s.naive.verdict === "BLOCK";
  $("naive-badge").innerHTML =
    `<span class="chip ${nb?"crit":"good"}">${nb?"BLOCKS THE MERGE":"SHIPS IT"}</span>`;
  $("naive-why").textContent = `${s.naive.describe} `
    + (nb ? `That is more than the ${pct(s.naive.threshold)} threshold, so the rule fires — with `
          + `no way to know whether the drop is meaningful.`
          : `That is within the ${pct(s.naive.threshold)} threshold, so the rule stays quiet. It `
          + `only ever sees the average.`);

  $("gate-badge").innerHTML = `<span class="chip ${vc}">${s.headline.toUpperCase()}</span>`;
  $("gate-why").textContent = s.body;

  $("note").className = s.agrees ? "note" : "note bad";
  $("note").innerHTML = s.agrees
    ? `<span class="dim">Both checks agree here. A gate that only ever disagreed with common
       sense would just be contrarian — the point is that it disagrees when it should.</span>`
    : `<strong>They disagree, and the conventional check is the one that is wrong.</strong>
       ${s.lesson}`;
}

/* ---------------- pipeline diagram ---------------- */
function pipeline(){
  const W = 1060, H = 216;
  const svg = el("svg", {viewBox:`0 0 ${W} ${H}`, role:"img",
    style:"width:100%;height:auto;display:block;overflow:visible"});
  svg.appendChild(el("title", {}, "AgentGate pipeline: suite, runner, agent, provider, "
    + "metrics, statistics, gate"));

  const stages = [
    ["Suite", "tasks + gold\ntrajectories", "#0f6e64"],
    ["Runner", "K reps,\nderived seeds", "#0f6e64"],
    ["Agent", "ReAct loop\n+ sandbox", "#0f6e64"],
    ["Provider", "live · cache\nreplay · mock", "#2a78d6"],
    ["Metrics", "42 scores\nper sample", "#eb6834"],
    ["Judge", "bias-controlled\nLLM grading", "#eb6834"],
    ["Statistics", "paired · clustered\nBH-adjusted", "#6b3fb5"],
    ["Gate", "3-way\nverdict", "#6b3fb5"],
  ];
  const boxW = 116, boxH = 74, gap = (W - stages.length*boxW) / (stages.length - 1);
  const y = 62;

  stages.forEach((st, i) => {
    const x = i * (boxW + gap);
    svg.appendChild(el("rect", {x, y, width:boxW, height:boxH, rx:9,
      fill:"var(--surface)", stroke:st[2], "stroke-width":"1.5"}));
    svg.appendChild(el("text", {x:x+boxW/2, y:y+25, "text-anchor":"middle",
      "font-size":"13", "font-weight":"640", fill:"var(--ink)"}, st[0]));
    st[1].split("\n").forEach((line, j) => {
      svg.appendChild(el("text", {x:x+boxW/2, y:y+43+j*13, "text-anchor":"middle",
        "font-size":"10.5", fill:"var(--ink-3)"}, line));
    });
    if (i < stages.length - 1){
      const x1 = x + boxW + 3, x2 = x + boxW + gap - 3;
      svg.appendChild(el("line", {x1, y1:y+boxH/2, x2, y2:y+boxH/2,
        stroke:"var(--axis)", "stroke-width":"1.5"}));
      svg.appendChild(el("path", {d:`M${x2-5},${y+boxH/2-3.5} L${x2},${y+boxH/2} `
        + `L${x2-5},${y+boxH/2+3.5}`, fill:"none", stroke:"var(--axis)", "stroke-width":"1.5"}));
    }
  });

  // Two-speed annotation
  const recEnd = 4*(boxW+gap) - gap/2;
  svg.appendChild(el("line", {x1:0, y1:32, x2:recEnd, y2:32, stroke:"#2a78d6",
    "stroke-width":"1.5"}));
  svg.appendChild(el("text", {x:recEnd/2, y:24, "text-anchor":"middle", "font-size":"11",
    fill:"#2a78d6", "font-weight":"600"}, "RECORD — hours, in the background"));
  svg.appendChild(el("line", {x1:recEnd, y1:32, x2:W, y2:32, stroke:"#0f6e64",
    "stroke-width":"1.5"}));
  svg.appendChild(el("text", {x:(recEnd+W)/2, y:24, "text-anchor":"middle", "font-size":"11",
    fill:"#0f6e64", "font-weight":"600"}, "DECIDE — seconds, in CI, offline"));

  // Cache loop under the provider
  const px = 3*(boxW+gap) + boxW/2;
  svg.appendChild(el("path", {d:`M${px},${y+boxH} L${px},${y+boxH+26} L${px+150},${y+boxH+26}`,
    fill:"none", stroke:"var(--axis)", "stroke-width":"1.2", "stroke-dasharray":"4 3"}));
  svg.appendChild(el("text", {x:px+8, y:y+boxH+42, "font-size":"10.5", fill:"var(--ink-3)"},
    "cached responses replay forever — CI never calls a model"));

  $("pipeline-svg").appendChild(svg);
}

/* ---------------- stack ---------------- */
function renderStack(){
  $("stack").innerHTML = D.stack.map(g => `
    <div class="sgroup">
      <h3>${g.title}</h3>
      <p>${g.blurb}</p>
      ${g.items.map(t => `
        <div class="tech">
          <div class="t"><b>${t.name}</b><code>${t.module.replace("agentgate.","")}</code></div>
          <div class="d">${t.what}</div>
        </div>`).join("")}
    </div>`).join("");
}

/* ---------------- model comparison ---------------- */
function renderModels(){
  const M = D.models;
  if (!M.models.length){ $("models-svg").remove(); return; }
  const colors = ["var(--s1)", "var(--s2)"];

  $("mlegend").innerHTML = M.models.map((m,i) =>
    `<span><i style="background:${colors[i]}"></i>${m.label}</span>`).join("");

  const W = 720, padL = 176, padR = 60, rowH = 46, top = 18;
  const H = top + M.rows.length*rowH + 26;
  const svg = $("models-svg");
  svg.innerHTML = "";
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const x = v => padL + v*(W-padL-padR);

  for (let t=0; t<=1.0001; t+=0.25){
    svg.appendChild(el("line", {x1:x(t), y1:top-6, x2:x(t), y2:top+M.rows.length*rowH,
      stroke:"var(--grid)", "stroke-width":"1"}));
    svg.appendChild(el("text", {x:x(t), y:H-8, "text-anchor":"middle", "font-size":"10.5",
      fill:"var(--ink-3)", "font-family":"var(--mono)"}, t.toFixed(2)));
  }

  M.rows.forEach((row, i) => {
    const yb = top + i*rowH;
    svg.appendChild(el("text", {x:padL-14, y:yb+rowH/2+4, "text-anchor":"end",
      "font-size":"12", fill:"var(--ink-2)"}, row.label));
    row.series.forEach((pt, j) => {
      if (!pt) return;
      const y = yb + 15 + j*15;
      const c = colors[j];
      if (pt.lo !== null){
        svg.appendChild(el("line", {x1:x(pt.lo), y1:y, x2:x(pt.hi), y2:y, stroke:c,
          "stroke-width":"2", opacity:"0.5", "stroke-linecap":"round"}));
        for (const v of [pt.lo, pt.hi])
          svg.appendChild(el("line", {x1:x(v), y1:y-3.5, x2:x(v), y2:y+3.5, stroke:c,
            "stroke-width":"1.8"}));
      }
      svg.appendChild(el("circle", {cx:x(pt.v), cy:y, r:"4.5", fill:c,
        stroke:"var(--surface)", "stroke-width":"2"}));
      const ttl = el("title", {});
      ttl.textContent = `${M.models[j].label} — ${row.label}: ${pt.v.toFixed(3)}`
        + (pt.lo===null ? "" : ` [${pt.lo.toFixed(3)}, ${pt.hi.toFixed(3)}]`);
      svg.appendChild(ttl);
    });
  });

  const head = `<thead><tr><th>Metric</th>${M.models.map(m =>
    `<th class="n">${m.label}</th>`).join("")}</tr></thead>`;
  const body = M.rows.map(r => `<tr><td>${r.label}</td>${r.series.map(p =>
    `<td class="n">${p ? p.v.toFixed(3) : "—"}</td>`).join("")}</tr>`).join("")
    + `<tr><td>Tokens spent</td>${M.tokens.map(t =>
        `<td class="n">${t===null?"—":t.toLocaleString()}</td>`).join("")}</tr>`;
  $("mtable").innerHTML = head + `<tbody>${body}</tbody>`;
}

renderStats();
renderTabs();
pipeline();
renderStack();
renderModels();
render();
"""

PAGE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgentGate — statistical regression gate for LLM agents</title>
<meta name="description" content="A CI gate that blocks a pull request only when an AI agent has
statistically significantly regressed. Paired non-inferiority testing, cluster-robust errors,
BH-FDR across 42 metrics, LLM-as-judge with bias controls, and an UNDERPOWERED verdict.">
<meta property="og:title" content="AgentGate — ship agent changes on evidence, not vibes">
<meta property="og:description" content="Statistical evaluation infrastructure for LLM agents:
paired non-inferiority testing, trajectory metrics, controlled LLM-as-judge, τ²-bench.">
<style>{STYLE}</style>
</head>
<body>
{BODY}
<script>{SCRIPT}</script>
</body>
</html>
"""


if __name__ == "__main__":
    build()
