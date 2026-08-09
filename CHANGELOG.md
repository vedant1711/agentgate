# Changelog

All notable changes to AgentGate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-08-09

First release. The gate works, the harness runs, and both have been exercised against real
models on a published benchmark rather than against a specimen.

### The second model, and the first result the project was built to produce

`qwen2.5:7b` recorded against the same 111 τ² tasks at K=3: **333 units, 317 completed, 3.83M
tokens, 5.4 hours.** Two models, identical tasks, identical seeds — so the comparison is paired.

The headline is that **the bigger model is not better, and the harness will not say that it is.**

| metric | llama3.2:3b | qwen2.5:7b |
|---|---|---|
| `outcome.task_success` | 0.180 [0.108, 0.252] | 0.111 [0.053, 0.169] |
| `judge.coherence` | 0.994 [0.988, 1.000] | **1.000 [1.000, 1.000]** |
| `judge.instruction_following` | **0.619 [0.559, 0.680]** | 0.356 [0.318, 0.395] |
| `trajectory.recall` | 0.535 [0.476, 0.595] | 0.442 [0.379, 0.505] |
| `trajectory.argument_correctness` | 0.145 [0.116, 0.173] | **0.321 [0.268, 0.374]** |
| `trajectory.loop_detected` | 0.000 | 0.033 [0.003, 0.063] |
| `efficiency.total_tokens` | 2,129 | 11,509 |
| `efficiency.latency_ms` | 13,260 | 58,702 |

**On task success, the leaderboard refuses to rank them** — the intervals overlap, so both sit in
tier 1. The paired test agrees and is more informative: delta −0.069 [−0.159, 0.021], 111 tasks,
10 better / 17 worse / 84 tied, p=0.14. Someone eyeballing 0.18 against 0.11 would report a 62%
relative difference; the suite's minimum detectable effect is 0.115, and the observed gap is 0.069.
**The gap is smaller than what this suite can resolve, and the honest answer is that we cannot
tell.** That is the entire project in one result.

Worth noting: the pair correlation is only **+0.059**. Pairing bought almost nothing here, because
two different model families succeed and fail on *different* tasks. Pairing is not a free win — it
pays when systems are related, and this measured it rather than assuming it.

**Three things the evidence does establish**, on non-overlapping intervals:

- **Instruction following is worse** on the bigger model (0.356 vs 0.619), while judged coherence
  is *perfect*. Coherence at 1.000 with task success at 0.111 is a metric proving its own
  uselessness as a discriminator.
- **Argument correctness is better** (0.321 vs 0.145). The bigger model fills arguments in more
  accurately while recovering fewer of the right tool calls — a genuinely mixed profile that a
  single blended score would have erased.
- **`safety.destructive_action_without_confirmation` fired on 79 samples across 27 tasks** for
  `qwen2.5:7b`, and was **skipped on all 333** for `llama3.2:3b`. Every destructive action the
  bigger model took was unconfirmed. Under the gate this is a SAFETY_FAIL, which bypasses the
  statistics entirely — no margin, no p-value, no discussion.

  The honest caveat: llama did not *avoid* unconfirmed destruction so much as barely act at all
  (2.0 LLM round trips against 5.4). A safety metric that never applies is not the same as a
  safety metric that passes, and the harness records it as skipped rather than as a zero for
  exactly that reason.

`agentgate versus` no longer dumps the raw analysis-units array — it reports the paired split,
the correlation, both one-sided p-values, and the detectable effect, then says in a sentence
whether the models were separated.

### Phase 8 — Parity with RAGAS and DeepEval

`docs/parity.md` states, metric by metric, exactly which AgentGate metric corresponds to which
RAGAS or DeepEval metric, and where the definitions deliberately differ — `rag.answer_relevancy`
skips RAGAS's synthetic-question round trip, and `answer_correctness` is deliberately *not*
implemented because blending factual overlap with semantic similarity hides which half moved.

**The page is explicit that the mapping is definitional and not numerically verified.** Both
libraries need an LLM for most metrics, which means a key and per-run cost, and this project's
constraint is that a clone runs at zero cost. "Compatible with RAGAS" is a claim people make
constantly and almost never test; the honest version names what remains unchecked.

To make it checkable rather than merely disclaimed, `agentgate export` writes a recorded run in
RAGAS's or DeepEval's input shape (or a neutral superset carrying cluster ids and seeds).
Verified against the real τ² run: 333 rows with real questions, answers and references. Tasks with
no declared reference export an empty ground truth so reference-requiring metrics skip the row,
rather than being graded against a fiction; and every repetition is exported, because collapsing
K to one row would destroy the per-sample comparison the export exists for.

Also fixed: the README's CI badge pointed at `agentgate/agentgate`, so it had been displaying an
unrelated project's build status.

### Phase 9 — The interactive demo

The Pages root is now a landing page whose centrepiece is a **gate you can operate**: drag the
non-inferiority margin and watch REGRESSION become UNDERPOWERED become PASS, on real data, with
the per-cluster differences plotted against the margin as it moves.

**Every slider position is evaluated at build time by the real engine**, through the same
`tests_at_margin()` the gate itself calls. The page looks values up; it does no statistics.
Reimplementing a paired t-test in browser JavaScript would have been easier and would have quietly
created a second, unverified implementation of the one thing this project must get right. A demo
that disagreed with the tool it demonstrates would be worse than no demo — so a test asserts the
grid and the gate rule identically at the policy's own margin and alpha.

Writing that test exposed a real gap: the policy's margin was not a reachable slider position, so
the page could show every verdict *except* the one the gate actually reached. The grid now snaps an
interior step to the policy margin — never an endpoint, because zero must stay reachable. "Set the
margin to zero and almost everything becomes UNDERPOWERED" is the demo's central lesson: proving
exact equality needs far more data than anyone has, which is precisely why the gate asks the
narrower question instead.

Four scenarios are shown, chosen to reach four different verdicts — nothing changed, a tool
vanished, output got wordier without getting worse, an injection payload leaked. Panels within a
scenario are ranked by **how many distinct verdicts appear across the grid**, not by effect size:
ranking by |delta| reliably surfaces token counts, whose deltas are in the hundreds and whose
verdicts never budge. A slider that cannot change the answer teaches nothing.

Two invariants are pinned by property tests: widening the margin can never make a verdict *more*
severe, and tightening alpha can never make REGRESSION easier to declare. A slider that
flip-flopped would mean the tests disagreed with themselves.

The report gallery moved to `gallery.html` to give the root to the demo. Both are still generated
from real pipeline output, never hand-written.

### Phase 8 — Documentation site

A real mkdocs-material site under `docs/`, deployed to `/docs/` on Pages so the report gallery
keeps the root. Architecture and the tiering rule are explained with mermaid diagrams rather than
prose alone.

**The results page is generated, never hand-written.** `agentgate docs results` renders
`docs/results.md` from the committed evidence snapshot, and `--check` fails CI when the two
disagree — the same guard already protecting the metric catalogue and the JSON Schemas. A results
page maintained by hand drifts from the results, and a stale number presented confidently is worse
than no number at all. Every generated table prints the interval beside the value.

CI now also builds the site with `--strict`, so a broken internal link fails the build instead of
shipping a dead page. It caught one immediately: `methodology.md` linked to a source file by
relative path, which resolves in the GitHub UI but not in a rendered site.

Two harness tests were pinning `qwen2.5:7b` as the example of a model with unmeasured throughput —
which recording that model would have falsified. They now construct an untimed card explicitly, so
measuring a real model can never break a test about unmeasured ones.

### Phase 8 — The first full baseline, and what it caught

`llama3.2:3b` recorded against all 111 τ² retail tasks at K=3: **333 units, 330 completed,
708,911 real tokens, 104 minutes.** Every number below is cluster-robust over 111 independent
clusters, which is why the intervals are narrow enough to be worth reading.

| metric | value | 95% CI |
|---|---|---|
| `outcome.task_success` | 0.180 | [0.108, 0.252] |
| `judge.coherence` | 0.994 | [0.988, 1.000] |
| `judge.instruction_following` | 0.619 | [0.559, 0.680] |
| `trajectory.recall` | 0.535 | [0.476, 0.595] |
| `trajectory.precision` | 0.574 | [0.517, 0.631] |
| `trajectory.argument_correctness` | 0.145 | [0.116, 0.173] |
| `trajectory.exact_match` | 0.000 | [0.000, 0.000] |
| `trajectory.error_recovery_rate` | 0.000 | [0.000, 0.000] |
| `trajectory.step_efficiency` | 0.766 | [0.711, 0.821] |
| `trajectory.loop_detected` | 0.000 | [0.000, 0.000] |

**Coherence 0.99 against task success 0.18 is the most valuable result the project has produced.**
The model is fluent, efficient, and non-looping — it does not flail, it does not loop, it wastes
almost nothing — and it is wrong four times in five. It calls roughly the right tools (recall 0.54)
with the wrong arguments (0.15) and never recovers from a tool error. Any evaluation resting on how
the output *reads* would score this model highly. That gap is the entire argument for the project.

**A defect the run exposed.** `outcome.json_valid` was scoring all 333 samples and failing 99% of
them — but τ² answers are prose. The metric had no `requires`, so it asked every suite a question
only some suites pose, and reported a category error as a catastrophic formatting failure. It now
requires `OUTPUT_SCHEMA`, the way a task declares it wants JSON, and is skipped otherwise. A golden
case pins the behaviour. This mattered beyond cosmetics: had that metric entered a gated family it
would have fired spuriously on every prose suite and consumed FDR budget from real findings.

`agentgate harness export` writes a compact snapshot to `results/harness.json`. The store itself is
not committed — tens of megabytes, regenerable from the cache — but the summary is, so every
recording session leaves a reviewable diff of what the project learned. Each value carries its
interval and its n; metrics that did not apply are omitted rather than zero-filled.

### Phase 8 — The living harness

AgentGate is meant to keep running, not to produce one report and stop. `agentgate.harness` is the
layer that makes that true: it tracks what has been measured, decides what to measure next under a
wall-clock budget, records it, and ranks the results without overclaiming.

**The leaderboard is the interesting part, because it is where this project could most easily
betray its own thesis.** Sorting models by point estimate and printing 1, 2, 3 asserts an ordering
between every adjacent pair — a large family of comparisons, each at no stated confidence, on
samples far too small to support them. So `agentgate leaderboard` ranks into **tiers**: a model
joins the current tier while its interval overlaps the tier leader's, and within a tier the answer
is *we cannot separate these*.

Two details make the tiering honest rather than decorative:

- **Overlap is compared against the tier leader, never the predecessor.** Chained overlap is not
  transitive: a can overlap b and b overlap c while a and c are disjoint, and predecessor-chaining
  would collapse all three into one tier that asserts a and c are indistinguishable. There is a
  test for exactly this.
- **The rule is conservative in a stated direction.** Non-overlapping intervals imply a real
  difference; overlapping ones do *not* imply the absence of one, because pairing removes the
  between-task variance that dominates both marginal intervals. So `agentgate versus` runs the
  proper paired test on the tasks both models ran, and the leaderboard's own verdict line points
  you there whenever a tier holds more than one model.

`agentgate trend` applies the same rule over time: movement is reported only when consecutive
intervals fail to overlap, and a suite whose content hash changed between two recordings breaks the
trend rather than having a line drawn through it — those scores answer different questions.

`agentgate harness next` plans work breadth-first, which is a statistical claim and not a
preference: an unmeasured model carries unbounded uncertainty, a partially measured one merely wide
uncertainty, so breadth removes more uncertainty per minute until every cell is touched once.
Models with no measured throughput are deferred rather than guessed at. The planner's own output is
the clearest argument for the cache/replay design — on this laptop `qwen3:4b` against the τ² suite
estimates at **27.8 hours**, while `llama3.2:3b` against the smoke suite is four minutes.

The ledger is **derived from the run store, not maintained alongside it**, so it cannot drift from
the runs it describes and there is no bookkeeping file to corrupt.

Fixed while building this: `RunStore(..., read_only=True)` created an empty file when the database
was absent, and DuckDB rejects a zero-byte database — the touch produced exactly the failure it was
meant to prevent. It now initialises a real, schema-bearing file.

### Phase 8 — Real models on a real benchmark

Two of the honest blockers in the Phase 7 write-up are now closed.

**A real model has touched it.** `agentgate models` reports which of eleven catalogued models are
reachable right now and why the rest are not — Ollama needs a daemon, everyone else needs a key —
so the harness never fails three minutes into a run over a missing credential. Local Ollama needs
no account at all, which is what makes the whole project runnable by anyone who clones it.

The measurements are the point, and they are unflattering in a useful way. `llama3.2:3b` scores
**4/8** on the smoke suite where the deterministic brain scores 8/8: it skips the customer lookup
and violates the refund-approval policy. Speed varies by an order of magnitude — 2.4s per
tool-calling turn for `llama3.2:3b` against ~75s for the reasoning model `qwen3:4b` — and a
70-task suite at K=4 is about 1,120 calls. That is the empirical case for the cache/replay
architecture: record slowly in the background, gate instantly from cache.

**The test set is no longer too small.** `suites/tau2_retail` adapts the retail domain of
[τ²-bench](https://github.com/sierra-research/tau2-bench) (MIT): 111 tasks over a 16-tool
environment with 500 users and 1,000 orders, graded against τ²'s own gold trajectories via a new
`trajectory_reference` checker. Because τ² tasks are mutually independent, each is its own
cluster — 111 against `crm_ops`'s 14, roughly eight times the effective sample size.

Three things were done deliberately rather than conveniently:

- **The adaptation is labelled, not hidden.** τ² is multi-turn with a simulated user; AgentGate is
  single-turn, so each scenario is flattened into one instruction carrying `known_info` while
  `unknown_info` is withheld. The suite description, the module docstring, and the dataset README
  all say scores here are not τ²-bench leaderboard scores.
- **`tau2_retail_agent` has no deterministic brain and refuses `--mode mock`.** A fabricated
  trajectory over real benchmark tasks would look authoritative and mean nothing.
- **The sandbox enforces the domain's real policies** — only pending orders are modifiable, and a
  cancellation reason must be one of τ²'s two allowed strings — so a policy violation is an
  observable tool error rather than a matter of opinion.

A test comparing the suite's demanded tools against the sandbox's implemented ones caught a real
gap: three gold trajectories call `get_item_details`, which had not been implemented. Without it
those tasks were unpassable by construction, and the model would have been blamed for a harness
defect. It is implemented now.

First real result: `llama3.2:3b` scores **0/4** on the first four τ² tasks, recovering 34% of the
gold tool calls (F1 0.44, argument correctness 0.16). It skips the `get_product_details`
verification steps and jumps straight to the exchange. That is a genuine, diagnosable weakness
measured on published tasks — exactly what the statistics exist to rule on.

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
