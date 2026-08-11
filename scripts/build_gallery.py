"""Build the example-report gallery.

The previous version listed eight cards labelled with fault names and verdicts, and assumed the
reader already knew what a "report" was, what a "faulted candidate" meant, and why
`UNDERPOWERED` was interesting. It read as a list of files.

A reader arriving here needs three things answered before a single link is worth clicking:

* **What is this thing I would be opening?** The report AgentGate posts on a pull request.
* **Why are there eight of them?** Because each is a different way an agent breaks, and the
  point is that the gate reaches a *different* verdict for each.
* **Which one should I look at first?** The two where the obvious answer is wrong.

So the page groups reports by what the gate decided, explains each verdict in plain language,
and leads with the two disagreements. The scenario descriptions come from the same copy the demo
uses, so the two pages cannot describe the same scenario differently.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from agentgate.demo import expected_verdict
from agentgate.report.chrome import STYLE, VERDICT_CLASS, footer, nav
from agentgate.report.story import copy_for, verdict_copy

SITE = Path("site")
REPORTS = SITE / "reports"

# Grouped by verdict so the page teaches the three-way rule rather than listing files. Ordered so
# the disagreements come first: those are the reports worth opening.
VERDICT_ORDER = ("SAFETY_FAIL", "UNDERPOWERED", "REGRESSION", "PASS")

VERDICT_INTRO: dict[str, str] = {
    "SAFETY_FAIL": (
        "The agent did something it must never do. These skip the statistics entirely — no "
        "average is good enough to make up for a security failure."
    ),
    "UNDERPOWERED": (
        "The numbers moved, but not by enough to tell a real change from ordinary variation. "
        "This is the verdict a threshold rule can never give, and the one that prevents false "
        "alarms."
    ),
    "REGRESSION": (
        "The agent really did get worse, by more than the tolerance the policy declared. These "
        "block the merge."
    ),
    "PASS": (
        "Non-inferiority was established — not merely 'we found no problem', but 'the suite was "
        "big enough to have found one and did not'."
    ),
}


def cards_for(verdict: str, reports: list[Path]) -> str:
    """Render one verdict group's cards, or nothing when no report reached that verdict."""
    matching = [r for r in reports if expected_verdict(r.stem) == verdict]
    if not matching:
        return ""

    cards = []
    for report in sorted(matching):
        copy = copy_for(report.stem)
        plain = verdict_copy(verdict)
        cards.append(
            f'  <a class="next-card" href="reports/{report.name}">\n'
            f'    <span class="tag {VERDICT_CLASS.get(verdict, "plain")}">'
            f"{plain['headline']}</span>\n"
            f'    <h3 style="margin:.6rem 0 .3rem;color:var(--ink)">{copy.title}</h3>\n'
            f'    <p class="dim" style="font-size:.88rem;margin:0">{copy.change}</p>\n'
            f"  </a>"
        )

    return (
        f'<section class="wrap">\n'
        f'  <span class="tag {VERDICT_CLASS.get(verdict, "plain")}">{verdict}</span>\n'
        f'  <h2 style="margin-top:.7rem">{verdict_copy(verdict)["headline"]}</h2>\n'
        f"  <p>{VERDICT_INTRO.get(verdict, '')}</p>\n"
        f'  <div class="next">\n' + "\n".join(cards) + "\n  </div>\n</section>\n"
    )


def build() -> Path:
    """Write ``site/gallery.html`` indexing whatever reports exist.

    The root is the demo; the gallery is one click in from it.
    """
    REPORTS.mkdir(parents=True, exist_ok=True)
    reports = list(REPORTS.glob("*.html"))

    if reports:
        groups = "".join(cards_for(verdict, reports) for verdict in VERDICT_ORDER)
    else:
        groups = (
            '<section class="wrap"><p class="dim">No reports were generated for this build. '
            "Run <code>agentgate demo --scenario dropped_tool --html report.html</code> to "
            "produce one.</p></section>\n"
        )

    page = PAGE.format(
        style=STYLE,
        nav=nav("gallery"),
        groups=groups,
        footer=footer(datetime.now(UTC).strftime("%B %Y")),
    )
    target = SITE / "gallery.html"
    target.write_text(page, encoding="utf-8")
    return target


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AgentGate &mdash; example reports</title>
<meta name="description" content="The report AgentGate posts on a pull request, for eight
different ways an AI agent can break.">
<style>{style}</style>
</head>
<body>
{nav}
<header class="hero wrap">
  <p class="eyebrow">Example reports</p>
  <h1>What AgentGate actually posts on your pull request</h1>
  <p class="lede">
    When the gate runs, it leaves a report like the ones below: the verdict, every metric with
    its error bars, what the statistics could and could not establish, and how much evidence it
    had to work with.
  </p>
  <p class="dim">
    Each report here comes from a real run against the same 70-task suite &mdash; a healthy agent
    against one that was deliberately broken in a specific, realistic way. Nothing is
    hand-written. Notice that the gate reaches a <strong>different verdict for each kind of
    breakage</strong>; that difference is the whole point.
  </p>
  <div class="row">
    <a class="btn primary" href="./">Start with the walkthrough</a>
    <a class="btn" href="docs/">Read the methodology</a>
  </div>
</header>

{groups}
<section class="wrap">
  <p class="eyebrow">How to read one</p>
  <h2>What you'll see inside</h2>
  <div class="next">
    <div class="card">
      <h3>A verdict, and why</h3>
      <p>Ship, block, or not enough evidence &mdash; with the reasoning, not just the label.</p>
    </div>
    <div class="card">
      <h3>Every metric with error bars</h3>
      <p>A number without a range is a guess. Each metric shows the interval it actually
      supports.</p>
    </div>
    <div class="card">
      <h3>What it could not tell you</h3>
      <p>The smallest change this suite was capable of detecting, so you know what a pass does
      and does not rule out.</p>
    </div>
  </div>
</section>

{footer}
</body>
</html>
"""


if __name__ == "__main__":
    print(f"wrote {build()}")
