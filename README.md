# AgentGate

**A statistical regression gate for AI agents.** The agent-world equivalent of a test suite
that fails the build — except it knows the difference between "your agent got worse" and
"your sample was small."

[![CI](https://github.com/agentgate/agentgate/actions/workflows/ci.yml/badge.svg)](https://github.com/agentgate/agentgate/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

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
| 1 | Provider layer (live/cache/replay/mock) | ⏳ |
| 2 | Reference agents, sandbox, fault knobs | ⏳ |
| 3 | Runner | ⏳ |
| 4 | Metrics engine | ⏳ |
| 5 | Judge subsystem | ⏳ |
| 6 | Statistics engine | ⏳ |
| 7 | Gate, CI, reporting, demo | ⏳ |
| 8 | Hardening and comparability | ⏳ |
| 9 | Public demo deployment | ⏳ |

## Quickstart

```bash
uv sync
uv run agentgate --help
```

## License

Apache-2.0.
