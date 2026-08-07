# Changelog

All notable changes to AgentGate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Phase 2 — Reference agents, sandbox, and fault injection

Added the systems under test and the regressions they suffer. `AgentUnderTest` is the whole
integration contract — one async `run(task, seed) -> Trajectory` — and three reference agents
implement it: `tool_agent` (ReAct loop over a sandboxed SQLite "company" with ten tools),
`rag_agent` (BM25 retrieval over a 120-document synthetic wiki), and `plan_agent`
(planner/executor whose retrieval is a tool call, so it exercises step efficiency and loop
detection). Success on the sandbox is checked tau-bench style against the **end state** — a
fluent apology that changed no rows is not a success — and business rules live in the sandbox,
so a policy violation is an observable tool error rather than a matter of opinion.

The eight fault knobs from F2 act *through* the system: `FAULT_PROMPT_DEGRADE` genuinely deletes
the policy paragraph, `FAULT_INJECTION_VULN` genuinely deletes the hardening paragraph, and
`FAULT_DROP_TOOL` genuinely removes a tool the agent still believes in — so the regression the
gate later catches is a real behavioural regression, not a hand-edited score. Each knob ships
with a declared metric signature (which metrics move, in which direction, and why), and a
parameterised test asserts no knob is inert.

The agents run on deterministic rule-based "brains" rather than canned strings, which is what
makes the whole pipeline offline, free, and byte-reproducible — and lets `cache` mode record
genuine replay fixtures without a single network call. OpenTelemetry spans follow OpenInference
conventions and are strictly optional (JSONL trajectories are the source of truth, so no metric
ever depends on a collector being up); `docker-compose.phoenix.yml` brings up a self-hosted
Phoenix for humans debugging a failing task.

### Phase 1 — Provider layer

Built the provider-agnostic L1 seam every model call flows through: a single `LLMClient`
composing four execution modes (`live`, `cache`, `replay`, `mock`), a write-through SQLite
response cache keyed by (model, messages, params, seed), per-provider token-bucket rate
limiting with concurrency caps, retry with full-jitter exponential backoff, and run-level
budget caps on requests, tokens, cost, and wall time. Recorded latency is replayed verbatim on
cache hits, because measuring SQLite lookup speed instead would make `efficiency.latency_ms`
meaningless in exactly the mode CI uses. Replay mode refuses to fall through to the network, so
a cache miss is a loud error rather than a silent quota charge; when 429s outlast the retry
budget the client degrades to replay for the rest of the run (E2 quota safety) instead of
turning CI red for a reason unrelated to the agent. The catalog carries free-tier limits and a
price table whose frontier entries exist purely so `efficiency.est_cost_usd` can project
enterprise cost from a run that cost nothing. Clocks, sleepers, and RNGs are injectable
throughout, so the whole backoff and rate-limit surface is asserted in 72 tests that never
sleep and never touch a network.

### Phase 0 — Repository foundation

Established the repository skeleton and the schema contract every later phase builds on: a uv
project on Python 3.12 with `ruff`, `mypy --strict`, `pytest`, and pre-commit wired into a
GitHub Actions workflow; Apache-2.0 licensing; and the full pydantic v2 model family —
`SuiteSpec`/`TaskSpec` (with reference trajectories supporting allowed-alternative tools,
optional steps, and unordered groups), `Trajectory` (a discriminated union of `llm_call`,
`tool_call`, `tool_result`, and `final` steps), `RunManifest` (whose `config_hash` deliberately
excludes wall-clock and host details so replayed runs on different machines hash identically),
`MetricResult`, `ComparisonResult`, `GatePolicy`, and `GateVerdict`. Aggregate numbers are
carried by an `Estimate` type that cannot be constructed without naming the interval method
used, which makes the "no number without its uncertainty" rule structural rather than
aspirational. JSON Schemas for all nine public models are exported to `schemas/` and checked in
CI, so the contract can never drift from the code.
