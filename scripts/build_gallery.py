"""Build the static report gallery landing page (K3).

GitHub Pages never sleeps, so this is a guaranteed-instant surface. It indexes whatever
reports the baseline workflow actually generated, so it can never advertise a report that
does not exist.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from agentgate.demo import expected_verdict
from agentgate.faults import SIGNATURES

SITE = Path("site")
REPORTS = SITE / "reports"

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgentGate — report gallery</title>
<style>
  :root {{ --bg:#fff; --fg:#1f2328; --muted:#57606a; --line:#d0d7de; --panel:#f6f8fa; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#0d1117; --fg:#e6edf3; --muted:#8b949e; --line:#30363d; --panel:#161b22; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
    font:16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif; }}
  .wrap {{ max-width:960px; margin:0 auto; padding:3rem 1.25rem 5rem; }}
  h1 {{ font-size:2rem; margin:0 0 .5rem; }}
  .lede {{ color:var(--muted); font-size:1.05rem; margin:0 0 2rem; }}
  .grid {{ display:grid; gap:1rem; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); }}
  a.card {{ display:block; text-decoration:none; color:inherit; background:var(--panel);
    border:1px solid var(--line); border-radius:10px; padding:1.1rem 1.2rem; }}
  a.card:hover {{ border-color:var(--fg); }}
  .verdict {{ display:inline-block; font-size:.72rem; font-weight:700; letter-spacing:.04em;
    padding:.15rem .5rem; border-radius:999px; color:#fff; margin-bottom:.6rem; }}
  .card h2 {{ font-size:1rem; margin:.1rem 0 .35rem; }}
  .card p {{ margin:0; color:var(--muted); font-size:.88rem; }}
  footer {{ margin-top:3rem; padding-top:1.25rem; border-top:1px solid var(--line);
    color:var(--muted); font-size:.85rem; }}
  code {{ background:var(--panel); padding:.1rem .35rem; border-radius:4px; font-size:.85em; }}
</style>
</head>
<body><div class="wrap">
<h1>AgentGate — report gallery</h1>
<p class="lede">
  Every report below was produced by the real pipeline: a baseline and a deliberately faulted
  candidate, run through agents, metrics, statistics, and the gate. Nothing here is hand-written.
  Each fault is a failure class that actually happens to agent systems — a prompt someone
  "simplified", a tool renamed in a refactor, a cost-driven model downgrade.
</p>

<div class="grid">
{cards}
</div>

<footer>
  Generated {generated}. Suite <code>crm_ops</code> — 70 tasks in 14 scenario clusters, run
  K=4 times each, entirely offline.
  Verdicts: <strong>PASS</strong> means non-inferiority was <em>proven</em>, not merely unproven
  bad. <strong>UNDERPOWERED</strong> means the suite genuinely cannot tell — the answer a naive
  threshold gate can never give.
</footer>
</div></body></html>
"""

COLOURS = {
    "PASS": "#1a7f37",
    "REGRESSION": "#cf222e",
    "SAFETY_FAIL": "#82071e",
    "UNDERPOWERED": "#9a6700",
    "ANY": "#57606a",
}


def build() -> Path:
    """Write ``site/gallery.html`` indexing whatever reports exist.

    The root is the interactive demo; the gallery is one click in from it.
    """
    REPORTS.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for report in sorted(REPORTS.glob("*.html")):
        scenario = report.stem
        signature = SIGNATURES.get(scenario)
        verdict = expected_verdict(scenario)
        colour = COLOURS.get(verdict, COLOURS["ANY"])
        description = (
            signature.simulates
            if signature
            else "The no-op control: an identical candidate, which must pass."
        )
        cards.append(
            f'  <a class="card" href="reports/{report.name}">\n'
            f'    <span class="verdict" style="background:{colour}">{verdict}</span>\n'
            f"    <h2>{scenario.replace('_', ' ')}</h2>\n"
            f"    <p>{description}</p>\n"
            f"  </a>"
        )

    if not cards:
        cards.append('  <p class="lede">No reports were generated for this build.</p>')

    page = _PAGE.format(
        cards="\n".join(cards),
        generated=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )
    target = SITE / "gallery.html"
    target.write_text(page, encoding="utf-8")
    return target


if __name__ == "__main__":
    print(f"wrote {build()}")
