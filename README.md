# AgentGate

**A statistical regression gate for AI agents.** The agent-world equivalent of a test suite
that fails the build — except it knows the difference between "your agent got worse" and
"your sample was small."

[![CI](https://github.com/vedant1711/agentgate/actions/workflows/ci.yml/badge.svg)](https://github.com/vedant1711/agentgate/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

**[Try the gate →](https://vedant1711.github.io/agentgate/)** · drag the margin and watch a verdict
change on real data · **[Docs](https://vedant1711.github.io/agentgate/docs/)** ·
**[Report gallery](https://vedant1711.github.io/agentgate/gallery.html)**

---

## The problem

Only ~5% of enterprise agentic AI projects reach production, and 55.4% of organizations name
*"AI agent reliability and hallucination management in production"* as their single biggest
adoption barrier. Meanwhile the industry-standard CI check for agents looks like this:

```python
if new_score < old_score - 0.03:  # "fail if any metric drops >3% on a 30-case golden set"
    fail_the_build()
```

On 30 cases, a 3-point drop is **noise**. That rule fires constantly on healthy agents and stays
silent on real regressions. It is a coin flip wearing a lab coat.

AgentGate replaces it with a **one-sided paired non-inferiority test** against a declared margin,
with Benjamini–Hochberg FDR control across the metric family, and — crucially — an
`UNDERPOWERED` verdict for when the suite genuinely cannot tell.

## What it does

- Runs each task **K times** (default 4) and reports `pass@1`, `pass@k`, and **`pass^k`** —
  the τ-bench reliability metric that exposes agents whose average looks fine and whose
  consistency does not.
- Scores **trajectories**, not just final answers: tool selection, ordering, argument
  correctness, step efficiency, loop detection, grounded reasoning — each as its own metric,
  never blended into one number that can hide a failure.
- Ships **error bars on everything**: CLT and Wilson intervals, cluster-robust SEs for suites
  whose tasks come from shared templates, BCa bootstrap for non-normal statistics.
- **Pairs** baseline against candidate on identical task instances and seeds, which turns the
  0.3–0.7 score correlation between two versions of an agent into free variance reduction.
- Treats the **judge as an instrument with error**: rubric-decomposed criteria, position-swap
  averaging, verbosity audits, judge/agent family independence, human-calibrated κ, and a frozen
  anchor set that detects judge drift between runs.
- Fails the gate **immediately** on any new safety tripwire — prompt-injection compliance, PII
  leak, forbidden tool call — with no statistics in the way.

## Status

Under active construction. See [CHANGELOG.md](CHANGELOG.md) for what has landed.

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository foundation, schemas, CLI skeleton | ✅ |
| 1 | Provider layer (live/cache/replay/mock) | ✅ |
| 2 | Reference agents, sandbox, fault knobs | ✅ |
| 3 | Runner | ✅ |
| 4 | Metrics engine | ✅ |
| 5 | Judge subsystem | ✅ |
| 6 | Statistics engine | ✅ |
| 7 | Gate, CI, reporting, demo | ✅ |
| 8 | Real models, τ²-bench, living harness, docs site | ✅ |
| 9 | Interactive public demo | ✅ |

## Quickstart

```bash
uv sync
uv run agentgate --help
uv run agentgate models          # which models are reachable right now, and why the rest are not
```

## The living harness

AgentGate keeps running. It records models against suites over time, into one growing store, and
answers the standing question: what do we know, how sure are we, and what should we measure next.

```bash
agentgate harness status                       # coverage: what has been measured, how completely
agentgate harness next --minutes 60            # what it would record next, and why
agentgate harness record --minutes 60          # record a session; resumable and interruptible
agentgate leaderboard --suite tau2_retail      # models ranked into separable tiers
agentgate versus --suite tau2_retail --baseline A --candidate B   # the paired test
agentgate trend --suite tau2_retail --model M  # how one model has moved over time
```

**The leaderboard reports tiers, not positions.** A model joins the current tier while its
confidence interval overlaps the tier leader's; within a tier the honest answer is *we cannot
separate these*. Ranking 1, 2, 3 by point estimate would assert an ordering between every adjacent
pair at no stated confidence — the exact error the gate exists to prevent in CI.

That rule is deliberately conservative: non-overlapping intervals imply a real difference, but
overlapping ones do not imply its absence. `agentgate versus` answers properly with the paired
test on the tasks both models ran, which is strictly more powerful.

`agentgate trend` applies the same standard over time, and refuses to draw a line across a suite
whose content hash changed — those scores answer different questions.

## Suites

| Suite | Tasks | Clusters | Ground truth | Agent |
|---|---|---|---|---|
| `suites/smoke` | 8 | 5 | Annotated goal state | `tool_agent` |
| `suites/crm_ops` | 70 | 14 | Annotated goal state | `tool_agent` |
| `suites/tau2_retail` | 111 | 111 | τ²-bench gold trajectories | `tau2_retail_agent` |

`tau2_retail` is a **single-turn adaptation** of the retail domain from
[τ²-bench](https://github.com/sierra-research/tau2-bench) (MIT). The tasks, the 16-tool
environment, and the gold trajectories are theirs; the interaction protocol is ours, so scores
here are *not* τ²-bench leaderboard scores. See [datasets/tau2/README.md](datasets/tau2/README.md).

It matters statistically: τ² tasks are mutually independent, so the suite contributes 111
clusters against `crm_ops`'s 14 — roughly eight times the effective sample size, which is what
moves the minimum detectable effect into a range where a real regression is actually detectable.

Running it needs a real tool-calling model. `tau2_retail_agent` has no deterministic brain and
**refuses to run in mock mode** on purpose: a fabricated trajectory over real benchmark tasks
would look authoritative and measure nothing.

```bash
ollama pull llama3.2:3b
uv run agentgate run --suite suites/tau2_retail --model ollama_chat/llama3.2:3b --mode cache
```

## License

Apache-2.0.
