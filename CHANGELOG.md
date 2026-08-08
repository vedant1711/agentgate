# Changelog

All notable changes to AgentGate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Phase 7 — Gate engine, reporting, CI, and the demo

The payoff. The gate applies C3's three-way rule — REGRESSION when a one-sided paired test
rejects beyond the margin, PASS when non-inferiority is *established* rather than merely
unrefuted, and UNDERPOWERED when the suite genuinely cannot tell — with Benjamini-Hochberg
control across the gated family and safety tripwires that bypass the statistics entirely.

Authored `suites/crm_ops`: 70 tasks in 14 scenario clusters (five paraphrases each, plus a
seeded injection subset), generated from what the healthy agent actually does so a reference can
never drift from correct behaviour by assumption. `agentgate demo --scenario X` runs a baseline
and a faulted candidate end to end offline and prints the verdict; `agentgate compare` does the
same from stored runs and exits with the verdict. Reports are a sticky PR comment and a
self-contained HTML page whose charts are inline SVG — a CI artifact opened offline still renders.
Three workflows: PR gate, baseline-on-main with a Pages gallery, and a budget-capped weekly canary.

Building the demo surfaced four real defects, each fixed rather than worked around:

- **The tests ignored clustering while the intervals respected it.** Tests now analyse per-cluster
  mean differences, so five paraphrases of one scenario count as one fact, not five — the
  anti-conservative direction E3 warns about.
- **Ratio margins never reached the tests.** `margin_ratio: 1.25` was resolved by the gate but
  the paired tests had been computed at margin 0, so the gate ruled on the wrong hypothesis.
  Margins are now resolved before the comparison is built.
- **Wilcoxon was answering the wrong question.** A non-inferiority margin is defined on the
  *mean*; Wilcoxon tests a median shift. On skewed differences the two genuinely disagree, and a
  candidate comfortably inside the margin was being failed. Margin tests now fall back to the
  sign-flip permutation test, which is distribution-free *and* about the mean.
- **Verdicts were not re-rulable.** The gate read pre-computed p-values, so re-evaluating at a
  different margin silently reused the old one. Comparisons now carry their analysis units and
  the gate recomputes both tests at its own margin — which is also what makes the demo's margin
  slider honest.

Also fixed: normal-approximation intervals on proportions no longer print bounds outside [0, 1].

### Phase 6 — Statistics engine

Part C in full, with every gate-critical formula implemented here rather than imported and every
docstring citing its source: CLT and Wilson intervals, cluster-robust sandwich SEs, BCa bootstrap
(resampling at cluster level when clusters exist), exact McNemar, paired t, Wilcoxon, sign-flip
permutation, Benjamini-Hochberg FDR, non-central-t power and MDE, the unbiased pass@k and pass^k
estimators, one-way variance decomposition with ICC, and law-of-total-variance propagation of
judge measurement error.

The analysis unit is the **task**, never the repetition — averaging K repetitions first is what
keeps intervals honest, since treating repetitions as independent tasks would shrink every
interval by sqrt(K) for nothing. Non-inferiority is implemented as a shift: testing `d_i + delta`
against zero is testing `mean(d) > -delta`, which is the hypothesis the gate actually cares
about, and both one-sided tests are computed so `INCONCLUSIVE` can be *expressed* rather than
silently collapsed into a pass.

The Monte-Carlo suite (2,000+ replications per case) is the credibility centrepiece and it
passes: type-I error lands inside [0.03, 0.07] at nominal 0.05, CLT/Wilson/paired coverage inside
[0.93, 0.97], empirical power within 5 points of the engine's prediction, pass^k mean-unbiased
where the naive plug-in is measurably biased, naive SEs shown collapsing to 76% coverage under
clustering while cluster-robust SEs repair it, and pairing's variance reduction growing
monotonically with correlation — E3's core claim reproduced on data where the correlation is set
rather than observed.

Two real bugs surfaced from property testing. Wilson's interval failed to contain its own point
estimate at p = 1 by one ULP, and BH's threshold rule disagreed with its own adjusted p-values at
the boundary; rejections are now derived from the adjusted values, because a report that prints
"adjusted p = 0.05, not rejected" at q = 0.05 is indefensible however correct its floating point.

`agentgate plan --target-mde` answers the suite author's question directly: the smoke suite's 8
tasks can detect 0.147 at 80% power, so a 0.03 target needs about 156.

### Phase 5 — Judge subsystem

The judge is now treated as an instrument with error rather than an oracle. Five anchored 1-5
rubrics, each its own criterion with elicited reasoning before the score and JSON-Schema-
constrained output; **J=3 draws** per item at temperature 0.3, with every draw kept so C5 can
fold the judge's own measurement variance into the metric's standard error. Malformed judge
output is retried and then *flagged* — never silently scored zero, which would turn a parsing
bug into a fabricated regression.

Bias is measured, not assumed away. Pairwise comparisons run in both slot orders and average;
their disagreement becomes a published **position-flip rate**. Verbosity and markdown-density
audits run on every judged suite, and a flagged verbosity correlation offers a length-controlled
re-analysis reported *beside* the raw scores, never instead of them. Judge/agent family
independence is default-deny, and the override attaches a SELF-JUDGING warning to every report
that used it.

Verification is the part that matters: the bias machinery is checked against synthetic judges
whose bias is *programmed*. A judge with an anchored 30% slot-A preference produces a measured
flip rate of 30%; a verbosity-loving judge trips the audit while an unbiased one does not; and
Cohen's kappa is checked against hand-computed examples including the 0/0 case the textbook
formula cannot express.

Calibration (`agentgate label`) measures judge-human agreement with kappa and Spearman rho, and
the 0.6 kappa floor is enforced: below it a criterion may be reported but may not back a gate. A
frozen anchor set detects judge drift between runs and banners it, and `agentgate.lock` pins
model, temperature, J, rubric hash, and anchor hash — reporting exactly which part of the ruler
moved rather than blocking the run.

### Phase 4 — Metrics engine

All 42 Part B metrics implemented as plugins behind one protocol: outcome, trajectory, RAG,
safety, and efficiency. The statistics engine never needs to know which is which — it reads
`dtype` and applies the machinery Part C says is legal for that measurement type.

The rule that shapes everything else: a metric whose requirements a task does not satisfy is
**skipped**, never scored zero. Scoring a missing reference as 0 would drag the suite mean down
and make a task-authoring gap look like an agent regression. Several metrics skip for subtler
reasons — `trajectory.error_recovery_rate` on a run with no errors is undefined, not perfect,
and scoring it 1.0 would make the suite mean move whenever the error *rate* changed.

The reference-trajectory matcher supports allowed-alternative tools, optional steps, and
unordered groups, and computes two alignments because the metrics need different things: an
ordered one for `exact_match`/`in_order_match`, and an unordered one for `any_order_match`,
precision/recall/F1, and argument correctness. `lcs_ratio` is the one metric awarding partial
order credit.

Judge-backed metrics run against a `Judge` protocol with a deterministic `LexicalJudge` default,
so faithfulness and grounded reasoning score offline for free — clearly named so a lexical
number is never mistaken for a rubric-judged one. The real bias-controlled judges arrive in
Phase 5. Similarity uses a local hashing embedder that is honest about being lexical rather than
semantic; the optional `embeddings` extra swaps in MiniLM.

Coverage: 126 hand-computed golden cases in committed YAML (every metric, ≥3 cases each,
including degenerate inputs), plus hypothesis property tests for bounds, the
`exact ⇒ in_order ⇒ any_order` implication chain, permutation invariance, and precision/recall/F1
consistency. Hypothesis found a real tokenizer bug along the way: the ASCII-only pattern was
silently mangling ordinary words ("café" → "caf"), now fixed to be Unicode-aware.
`docs/metrics.md` is generated from the registry and checked in CI.

### Phase 3 — Runner

Added the layer that turns "run this agent" into `N tasks × K repetitions` of reproducible
evidence: a suite loader that refuses structurally broken YAML and *warns* about statistically
dubious suites (singleton clusters, missing reference trajectories, a task count too small for
any interval to mean anything), a scheduler with bounded asyncio concurrency and deterministic
per-unit seeds derived from `(base_seed, task_id, rep)`, resume-from-partial via JSONL flushed
as each unit lands, and DuckDB persistence of runs and **per-sample** records — aggregates
cannot be re-analysed at a different K or margin, and re-analysis is the whole point of the
interactive demo.

Two decisions are worth naming. Run ids are *derived* from the suite hash, agent, K, seed, mode,
and active faults rather than generated, so resume needs no state file and re-running the same
config is a no-op instead of a duplicate. And `efficiency.latency_ms` now scores the sum of
attributed step durations rather than wall-clock: wall-clock can never replay identically, so a
gate on it would fire on scheduling noise. Wall time is still recorded, in a field explicitly
excluded from the analysis payload — which is now a first-class concept (`analysis_payload` /
`analysis_digest`) defining exactly what AgentGate promises to reproduce.

Pairing guards live here too: `assert_comparable` refuses (never warns) when two runs differ in
suite content, K, or seed, because a paired test across different task instances is not a paired
test at all.

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
