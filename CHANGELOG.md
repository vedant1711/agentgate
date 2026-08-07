# Changelog

All notable changes to AgentGate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
