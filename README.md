# AgentGate

**Your AI agent scored 79% this week and 100% last week. Did you break it, or did it just have a
bad day?**

AgentGate is a CI check that answers that question with statistics instead of a guess — and says
so out loud when it genuinely cannot tell.

[![CI](https://github.com/vedant1711/agentgate/actions/workflows/ci.yml/badge.svg)](https://github.com/vedant1711/agentgate/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![Release](https://img.shields.io/github/v/release/vedant1711/agentgate)](https://github.com/vedant1711/agentgate/releases)

### 👉 **[Try the interactive demo](https://vedant1711.github.io/agentgate/)** — walk through real cases where the obvious answer is wrong

[Documentation](https://vedant1711.github.io/agentgate/docs/) ·
[Architecture](https://vedant1711.github.io/agentgate/docs/architecture/) ·
[Example reports](https://vedant1711.github.io/agentgate/gallery.html) ·
[Results](https://vedant1711.github.io/agentgate/docs/results/)

---

## The problem

You have an agent that books flights, or answers support tickets, or writes code. You change one
line of its system prompt. You rerun your 30-case test set.

**It went from 27/30 to 24/30. Ship it or not?**

Nobody knows. Agents are stochastic — rerun the *identical* agent and you'd also get a different
number. So the industry-standard CI check is this:

```python
if new_score < old_score - 0.03:  # "fail if any metric drops >3% on a golden set"
    fail_the_build()
```

On 30 cases, a 3-point drop is **noise**. That rule fires constantly on healthy agents and stays
silent on real regressions. Within two weeks the team mutes it, and then nothing is checked at all.

## Why it's genuinely hard

It isn't just that the rule is too twitchy. **It's wrong in both directions**, and here are two
real runs from this repo that prove it:

| What happened | Score | A normal CI check says | AgentGate says |
|---|---|---|---|
| A support ticket contained a hidden instruction, and the agent obeyed it | 100% → **100%** | ✅ ship it | 🛑 **security failure** |
| Someone asked the agent to be more thorough | 100% → **79%** | ❌ block the merge | ⚠️ **not enough evidence** |

The first is a **prompt-injection breach with a perfect scorecard** — the average never moved, so
a threshold check waves it through. The second is a **21-point drop that isn't established** —
blocking on it would be a false alarm that teaches everyone to ignore the gate.

Both come from real pipeline runs. You can step through them in the
[interactive demo](https://vedant1711.github.io/agentgate/).

## How AgentGate answers it

**1 · Run both sides on identical tasks.** Same test cases, same random seeds, before and after.
Task difficulty varies enormously, and pairing cancels it out — so what's left is the effect of
your change, not the luck of the draw.

**2 · Compare per task, not totals.** Two totals can match while every individual case moved.
Looking at each task's before-and-after is what separates a real pattern from a few coincidences.

**3 · Answer three ways, not two.**

| Verdict | Meaning |
|---|---|
| ✅ `PASS` | Non-inferiority was **established** — the suite was big enough to catch a regression of the size you declared you'd care about, and didn't find one. |
| ❌ `REGRESSION` | It really did get worse, by more than your declared tolerance. Block the merge. |
| ⚠️ `UNDERPOWERED` | Not enough evidence to tell a real change from ordinary variation. **This is not a pass.** |
| 🛑 `SAFETY_FAIL` | The agent did something it must never do. Statistics are skipped entirely. |

`UNDERPOWERED` is the verdict no threshold rule can ever give, and it's the one that keeps the
gate honest on small test sets. Instead of a fake green tick, you get the **minimum detectable
effect** — a number telling you what your suite *could* have caught.

Underneath: cluster-robust standard errors so five rewordings of one scenario count once,
Benjamini–Hochberg correction because checking 40 metrics at α=0.05 false-alarms ~87% of the time
by chance, BCa bootstrap for non-normal statistics, and LLM-judge scores treated as measurements
with their own error bars. [Full methodology →](https://vedant1711.github.io/agentgate/docs/methodology/)

## Quickstart

```bash
uv sync
uv run agentgate --help

# See the gate decide, end to end, offline and free — no API key needed
uv run agentgate demo --scenario dropped_tool --html report.html

# Which models can this machine actually reach right now, and why not the rest?
uv run agentgate models
```

Every command above runs with **no network, no API key, and no cost.** That's a hard constraint of
the project, not a demo mode.

To drive it with a real model:

```bash
ollama pull llama3.2:3b
uv run agentgate run --suite suites/tau2_retail --model ollama_chat/llama3.2:3b --mode cache
```

## Real results

Two open models on 111 tasks from [τ²-bench](https://github.com/sierra-research/tau2-bench), a
published agent benchmark. 666 runs, 4.5M tokens, ~7 hours of real inference:

| | Llama 3.2 3B | Qwen2.5 7B |
|---|---|---|
| **Finished the job correctly** | 0.180 `[0.108, 0.252]` | 0.111 `[0.053, 0.169]` |
| Followed the instructions | **0.619** `[0.559, 0.680]` | 0.356 `[0.318, 0.395]` |
| Sounded coherent | 0.994 | **1.000** |
| Passed the right arguments | 0.145 `[0.116, 0.173]` | **0.321** `[0.268, 0.374]` |
| Tokens spent | 2,129 | 11,509 |

**Read rows 1 and 3 together.** The bigger model sounds *more* coherent while finishing *fewer*
jobs. That's exactly why "it reads well" is not a measurement.

And on finishing the job, **AgentGate refuses to rank them.** Eyeballing 0.180 vs 0.111 gives a
62% relative difference — but paired across 111 tasks the difference is −0.069 `[−0.159, 0.021]`,
and this suite's minimum detectable effect is 0.115. The gap is smaller than what the evidence can
resolve, so the honest answer is *we cannot tell*.

[Full numbers with error bars →](https://vedant1711.github.io/agentgate/docs/results/)

## What's inside

| Layer | What it owns |
|---|---|
| **Suites** | Tasks + gold trajectories, content-hashed so a changed suite is never silently compared |
| **Provider** | `live` / `cache` / `replay` / `mock` — CI uses `replay`, so it never calls a model |
| **Agents** | ReAct loops over a 10-tool mock company, plus a τ²-bench adapter and 8 fault-injection knobs |
| **Metrics** | 42 metrics — outcome, trajectory, RAG, safety, efficiency, judge — every one per-sample |
| **Judge** | Position-swap averaging, verbosity audits, family independence, κ calibration, drift detection |
| **Statistics** | Paired tests, cluster-robust SEs, BH-FDR, BCa bootstrap, power/MDE |
| **Gate** | The three-way rule, declared policy, safety tripwires that bypass statistics |
| **Harness** | Tiered leaderboards, trends, and a committed evidence snapshot that grows over time |

[Architecture, with diagrams →](https://vedant1711.github.io/agentgate/docs/architecture/)

## Suites

| Suite | Tasks | Clusters | Ground truth | Agent |
|---|---|---|---|---|
| `suites/smoke` | 8 | 5 | Annotated goal state | `tool_agent` |
| `suites/crm_ops` | 70 | 14 | Annotated goal state | `tool_agent` |
| `suites/tau2_retail` | 111 | 111 | τ²-bench gold trajectories | `tau2_retail_agent` |

`tau2_retail` is a **single-turn adaptation** of the retail domain from τ²-bench (MIT). The tasks,
the 16-tool environment, and the gold trajectories are theirs; the interaction protocol is ours,
so these are **not** τ²-bench leaderboard scores. See [datasets/tau2/README.md](datasets/tau2/README.md).

It matters statistically: τ² tasks are mutually independent, so the suite contributes 111 clusters
against `crm_ops`'s 14 — roughly eight times the effective sample size.

`tau2_retail_agent` has no deterministic brain and **refuses to run in mock mode** on purpose: a
fabricated trajectory over real benchmark tasks would look authoritative and measure nothing.

## The living harness

AgentGate keeps running. It records models against suites over time into one growing store, and
answers the standing question: what do we know, how sure are we, and what's most worth measuring
next?

```bash
agentgate harness status               # what's been measured, and how completely
agentgate harness next --minutes 60    # what it would record next, and why
agentgate harness record --minutes 60  # record a session; resumable and interruptible
agentgate leaderboard --suite tau2_retail
agentgate versus --suite tau2_retail --baseline A --candidate B
agentgate trend --suite tau2_retail --model M
```

**The leaderboard reports tiers, not positions.** A model joins a tier while its interval overlaps
the tier leader's; within a tier the honest answer is *we cannot separate these*. Printing 1, 2, 3
by point estimate would assert an ordering between every adjacent pair at no stated confidence —
the exact error the gate exists to prevent.

Every recording session commits an updated evidence snapshot, so the repository's git history *is*
the history of what the project has learned.
[How it works →](https://vedant1711.github.io/agentgate/docs/harness/)

## Status

`v0.1.0` — all ten build phases complete. See [CHANGELOG.md](CHANGELOG.md).

Python 3.12 · `mypy --strict` · 764 tests · pydantic v2 · DuckDB · LiteLLM ·
OpenTelemetry/OpenInference tracing to Phoenix · Apache-2.0.

## Honest limitations

The suites are small next to production traffic. The judge is itself a model with measured biases.
The τ² adaptation is single-turn where the original is multi-turn. The RAGAS/DeepEval parity
mapping is definitional and has not been numerically verified.

All of it is stated plainly in [Limitations](https://vedant1711.github.io/agentgate/docs/limitations/)
— which is the same discipline as the `UNDERPOWERED` verdict, applied to the project itself.

## License

Apache-2.0.
