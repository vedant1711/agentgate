# Limitations

What AgentGate does not tell you, stated plainly. A tool that quantifies uncertainty and then
hides its own would be self-refuting.

---

## The conclusions are about *your suite*, not about your agent

Every interval here is an estimate of a population mean **over the task distribution your suite
samples**. "Task success fell by 4 points [1.2, 6.8]" means *on tasks like these*. If your suite
over-represents refunds and your users mostly ask about shipping, the gate is measuring
something adjacent to what you care about.

Nothing in the statistics can detect this. Distribution shift between the suite and production is
a suite-authoring problem, and the only defence is writing tasks that look like the traffic.

## `UNDERPOWERED` is common, and that is the honest answer

A 30-task suite with per-task noise around 0.15 needs roughly 150 tasks to detect a 3-point
regression at 80% power. Most suites people actually have cannot detect the margins people
actually configure. AgentGate says so instead of returning a green tick.

The fix is more tasks, more repetitions, or a wider margin — `agentgate plan` tells you which and
how many. It is not a bug that the gate declines to rule.

## Judge validity is bounded by κ, and κ is bounded by your labelling

- A judge below κ = 0.6 may inform a report but may not gate. That threshold is a convention, not
  a law.
- κ is itself imprecise below ~30 labelled items; the calibration report warns about this.
- κ measures agreement with *your* labeller. A systematically wrong human produces a
  systematically wrong judge with excellent agreement.
- The anchor set detects judge *drift*, not judge *error*. A judge that was wrong from the start
  and stays wrong shows no drift at all.

## The default embedder is lexical, not semantic

`outcome.semantic_similarity` and `rag.answer_relevancy` use a deterministic hashed
bag-of-words encoder by default. It catches paraphrase-by-word-choice and misses
paraphrase-by-meaning. It is the default because it needs no download and produces byte-identical
vectors on every machine, which is what reproducible CI requires — but it is named
`hashing-bow` in every report for a reason.

Install the `embeddings` extra to swap in MiniLM. Any published similarity number should say
which encoder produced it.

## The default judge is not an LLM

Without a configured judge, judge-backed metrics fall back to `LexicalJudge`, which splits text
into sentences and scores support by word overlap. It is honest about being a deterministic
proxy, and reports name it — but a `lexical` faithfulness score is not a rubric-judged one.

Claim attribution is also **per-passage**: a claim must be supported by a single retrieved
passage. A claim that is only supported by combining two passages scores as unsupported.

## Free-tier throughput sets a ceiling

Live runs are shaped to conservative rate limits (roughly 14–28 requests/minute). A 200-task
suite at K=4 with a 6-step agent is around 4,800 model calls — hours on a free tier. This is why
`replay` is the CI default and `cache` is the recording mode; live mode is for the weekly canary,
not for the gate.

## Reproducibility has a defined scope

Two runs with the same manifest and the same replay cache produce identical **analysis payloads**
— trajectories minus `started_at`, `ended_at`, and `wall_ms`. Those three fields cannot replay
identically and never affect a score. `efficiency.latency_ms` is the sum of *attributed step
durations*, not wall-clock, for exactly this reason.

Live mode is not reproducible at all, and does not claim to be.

## The reference agents are specimens, not products

`tool_agent`, `rag_agent`, and `plan_agent` run on deterministic rule-based policies in mock
mode. They exist so the statistics and gate layers have something honest to chew on for free, and
so the fault knobs have somewhere to bite. They are not a claim about how real agents behave.

The demo verdicts are real verdicts on real trajectories — but the *agent* producing them is a
specimen.

## Statistical caveats worth naming

- **Cluster-level analysis reduces effective n.** A 70-task suite in 14 clusters has 14
  independent observations, not 70. That is correct, and it is also why clustered suites need
  more clusters rather than more paraphrases.
- **BCa can fail on degenerate samples** and falls back to the percentile interval, recorded in
  the method label.
- **`trajectory.error_recovery_rate` has a selection effect**: only trajectories that had errors
  contribute, so its denominator moves with the error rate.
- **`safety.*` metrics are detectors, not proofs.** A canary that was not leaked does not mean no
  PII was leaked; it means that canary was not. Absence of evidence.
- **McNemar's odds ratio is infinite** when the baseline had no discordant successes. Reported as
  such rather than smoothed.

## What is deliberately not implemented

- Multi-turn user simulation (τ-bench's interactive component).
- Cross-suite meta-analysis, or pooling evidence across suite versions. The gate refuses
  cross-version comparison rather than attempting to reconcile it.
- Automatic suite generation. Tasks are authored; the generator in `scripts/` builds *references*
  from observed correct behaviour, not tasks from thin air.
- Sequential testing / early stopping. Every comparison analyses a complete run, so the p-values
  do not need alpha-spending correction — and adding sequential analysis without that correction
  would silently inflate the error rate.
