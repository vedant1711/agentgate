"""Shared page chrome for the published site: one palette, one nav, one voice.

The demo and the report gallery were built by separate scripts with separate stylesheets, and it
showed — different colours, different type, and no way to get from one to the other. A reader who
landed on the gallery had no route back and no reason to believe the two pages belonged to the
same project.

Keeping the styles and the navigation here means the pages cannot drift apart: changing a colour
changes it everywhere, and adding a destination adds it to every page's nav at once. The mkdocs
site brings its own theme, so it is linked to rather than restyled — but it appears in the same
nav, in the same order, on every page.
"""

from __future__ import annotations

from typing import Final

SITE_ROOT: Final = "https://vedant1711.github.io/agentgate/"
REPO_URL: Final = "https://github.com/vedant1711/agentgate"

STYLE: Final = """
  :root {
    --paper:#fcfbf9; --ink:#12151b; --dim:#59626f; --line:#e0dcd4; --panel:#f4f1eb;
    --raised:#ffffff;
    --accent:#0f6e64; --accent-soft:#d9ece9; --on-accent:#ffffff;
    --good:#0a7040; --good-soft:#dcf0e4;
    --bad:#b03028; --bad-soft:#fbe4e2;
    --warn:#a5620a; --warn-soft:#fbeeda;
    --stop:#6b3fb5; --stop-soft:#ece4fa;
    color-scheme: light;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --paper:#0d1014; --ink:#e9ecef; --dim:#98a2b0; --line:#232932; --panel:#151a20;
      --raised:#1a2029;
      --accent:#4fd6c4; --accent-soft:#0d2f2b; --on-accent:#0d1014;
      --good:#5ed69a; --good-soft:#0e2a1d;
      --bad:#ff9187; --bad-soft:#2e1512;
      --warn:#f0b249; --warn-soft:#2d2210;
      --stop:#bda6f5; --stop-soft:#221a35;
      color-scheme: dark;
    }
  }
  :root[data-theme="dark"] {
    --paper:#0d1014; --ink:#e9ecef; --dim:#98a2b0; --line:#232932; --panel:#151a20;
    --raised:#1a2029;
    --accent:#4fd6c4; --accent-soft:#0d2f2b; --on-accent:#0d1014;
    --good:#5ed69a; --good-soft:#0e2a1d;
    --bad:#ff9187; --bad-soft:#2e1512;
    --warn:#f0b249; --warn-soft:#2d2210;
    --stop:#bda6f5; --stop-soft:#221a35;
    color-scheme: dark;
  }

  * { box-sizing:border-box; }
  html { -webkit-text-size-adjust:100%; scroll-behavior:smooth; }
  body { margin:0; background:var(--paper); color:var(--ink);
    font:16px/1.65 ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    font-variant-numeric:tabular-nums; }
  .wrap { max-width:900px; margin:0 auto; padding:0 1.25rem; }

  h1,h2,h3 { font-family:ui-serif,Georgia,"Times New Roman",serif; font-weight:600;
    text-wrap:balance; letter-spacing:-.015em; margin:0 0 .7rem; }
  h1 { font-size:clamp(2rem,5vw,3.1rem); line-height:1.1; }
  h2 { font-size:clamp(1.35rem,2.8vw,1.85rem); }
  h3 { font-size:1.02rem; font-family:inherit; font-weight:650; margin-bottom:.3rem; }
  p { margin:0 0 1rem; max-width:64ch; }
  .dim { color:var(--dim); }
  code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.86em;
    background:var(--panel); padding:.1rem .3rem; border-radius:4px; }
  a { color:var(--accent); text-underline-offset:.18em; }
  .eyebrow { font-size:.7rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
    color:var(--accent); margin:0 0 .8rem; }

  .topbar { position:sticky; top:0; z-index:20; border-bottom:1px solid var(--line);
    background:var(--paper); display:flex; align-items:center; gap:1.5rem; flex-wrap:wrap;
    padding:.75rem max(1.25rem,calc((100% - 900px)/2)); }
  .brand { font-family:ui-serif,Georgia,serif; font-weight:650; font-size:1.05rem;
    text-decoration:none; color:var(--ink); }
  .topbar .links { display:flex; gap:1.1rem; margin-left:auto; flex-wrap:wrap; }
  .topbar .links a { font-size:.88rem; text-decoration:none; color:var(--dim); }
  .topbar .links a:hover { color:var(--ink); }
  .topbar .links a.here { color:var(--accent); font-weight:600; }

  section { padding:3.5rem 0; border-top:1px solid var(--line); }
  header.hero { padding:4rem 0 3rem; }
  .lede { font-size:clamp(1.05rem,2vw,1.25rem); color:var(--dim); max-width:56ch; }

  .btn { display:inline-block; padding:.62rem 1.15rem; border-radius:7px; text-decoration:none;
    border:1px solid var(--line); color:var(--ink); font-size:.92rem; font-weight:550; }
  .btn.primary { background:var(--accent); border-color:var(--accent); color:var(--on-accent); }
  .btn:hover { border-color:var(--accent); }
  .row { display:flex; flex-wrap:wrap; gap:.7rem; margin-top:1.8rem; }

  .tag { display:inline-flex; align-items:center; padding:.22rem .62rem; border-radius:999px;
    font-size:.72rem; font-weight:700; letter-spacing:.03em; }
  .tag.ship { background:var(--good-soft); color:var(--good); }
  .tag.block { background:var(--bad-soft); color:var(--bad); }
  .tag.warn { background:var(--warn-soft); color:var(--warn); }
  .tag.stop { background:var(--stop-soft); color:var(--stop); }
  .tag.plain { background:var(--panel); color:var(--dim); }

  .card { border:1px solid var(--line); border-radius:10px; padding:1.15rem;
    background:var(--raised); }
  .card p { font-size:.88rem; color:var(--dim); margin:0; }

  .next { display:grid; gap:1rem; grid-template-columns:repeat(auto-fit,minmax(235px,1fr));
    margin-top:1.4rem; }
  a.next-card { display:block; text-decoration:none; color:inherit; border:1px solid var(--line);
    border-radius:10px; padding:1.2rem; background:var(--raised); }
  a.next-card:hover { border-color:var(--accent); }
  a.next-card h3 { color:var(--accent); }

  table { width:100%; border-collapse:collapse; font-size:.9rem; }
  th,td { text-align:left; padding:.55rem .6rem; border-bottom:1px solid var(--line); }
  th { font-size:.72rem; letter-spacing:.07em; text-transform:uppercase; color:var(--dim); }
  td.num, th.num { text-align:right; font-family:ui-monospace,monospace; }
  .scroll { overflow-x:auto; }

  footer { padding:2.5rem 0 4rem; border-top:1px solid var(--line); font-size:.85rem;
    color:var(--dim); }
"""

VERDICT_CLASS: Final[dict[str, str]] = {
    "PASS": "ship",
    "REGRESSION": "block",
    "UNDERPOWERED": "warn",
    "SAFETY_FAIL": "stop",
    "INCONCLUSIVE": "plain",
}
"""Maps a gate verdict to the palette token that carries its meaning."""


def nav(here: str) -> str:
    """Render the site navigation, marking the current page.

    Args:
        here: One of ``demo``, ``docs``, or ``gallery``.

    Returns:
        The nav markup. Links are relative to the site root, so every page must be published at
        the depth the paths assume.
    """
    items = (
        ("demo", "./", "Demo"),
        ("docs", "docs/", "Documentation"),
        ("gallery", "gallery.html", "Example reports"),
    )
    links = "".join(
        f'    <a href="{href}"{" class='here'" if key == here else ""}>{label}</a>\n'
        for key, href, label in items
    )
    return (
        '<nav class="topbar">\n'
        '  <a class="brand" href="./">AgentGate</a>\n'
        '  <div class="links">\n'
        f"{links}"
        f'    <a href="{REPO_URL}">GitHub</a>\n'
        "  </div>\n"
        "</nav>\n"
    )


def footer(generated: str) -> str:
    """The shared page footer."""
    return (
        '<footer class="wrap">\n'
        f'  <p>Built by <a href="https://github.com/vedant1711">vedant1711</a> &middot; '
        f"Apache-2.0 &middot; generated {generated} &middot; "
        '<a href="docs/limitations/">limitations</a></p>\n'
        '  <p style="font-size:.8rem;max-width:60ch">Read the limitations page before trusting '
        "any number here. The test suites are small next to production traffic, and the judge is "
        "itself a model with measured biases. Everything that applies is stated rather than "
        "omitted.</p>\n"
        "</footer>\n"
    )
