# Parity with RAGAS and DeepEval

AgentGate reimplements metrics that already exist elsewhere. That deserves justification, and it
deserves a precise statement of what each one is — otherwise "answer relevancy" is just a word, and
a number labelled with it is unfalsifiable.

This page states the correspondence exactly, including where AgentGate deliberately differs.

!!! warning "What has and has not been verified"

    The mappings below are **definitional**: they were established by reading each library's
    documented formula and comparing it to the implementation here. AgentGate's own values are
    pinned by hand-computed golden cases in `tests/fixtures/metrics/`.

    A **numerical** side-by-side against RAGAS and DeepEval has **not** been run. Both libraries
    require an LLM for most of their metrics, which means an API key and per-run cost, and this
    project's hard constraint is that a clone must be fully runnable at zero cost. Running that
    comparison is described at the bottom of this page and is left to anyone who wants it.

    Saying this plainly is the point. "Compatible with RAGAS" is a claim people make constantly
    and almost never test; the honest version is "here is the formula, here is ours, here is
    exactly what remains unchecked."

---

## Why reimplement at all

Three reasons, in order of weight.

**Uncertainty is not optional here.** RAGAS and DeepEval return point estimates. This project's
entire premise is that a point estimate without an interval cannot support a merge decision, so
every metric must expose per-sample values that the statistics layer can pair, cluster, and
bootstrap. Wrapping a library that returns one number per run would have made the gate impossible.

**Determinism in CI.** The gate must produce the same verdict from the same inputs, on a machine
with no network. That requires every metric to be computable from a recorded trajectory in replay
mode. Metrics that call an LLM internally, with their own prompts and their own retry behaviour,
cannot offer that.

**Judge variance must be visible.** When a score comes from a model, its variance belongs in the
standard error of anything computed from it. That requires access to the individual judge draws,
not a collapsed mean.

---

## RAG metrics

| AgentGate | RAGAS equivalent | Relationship |
|---|---|---|
| `rag.context_precision` | `context_precision` | **Same quantity, different judge.** Fraction of retrieved contexts that are relevant. RAGAS decides relevance with an LLM; AgentGate uses the task's declared gold context ids where the suite provides them, falling back to the judge otherwise. Ground truth beats a model's opinion when ground truth exists. |
| `rag.context_recall` | `context_recall` | **Same quantity.** Fraction of the gold contexts that were retrieved. Computed from declared ids, so it needs no model at all. |
| `rag.answer_relevancy` | `answer_relevancy` | **Related, deliberately simpler.** RAGAS generates *n* synthetic questions from the answer and measures their mean cosine similarity to the original question. AgentGate scores the answer against the question directly. RAGAS's round trip is a reasonable idea, but it multiplies the number of model calls by *n* and introduces a second generation step whose variance is not reported. |
| `rag.faithfulness` | `faithfulness` | **Same decomposition.** Claims are extracted from the answer and each is checked for support in the retrieved context. AgentGate keeps the per-claim verdicts so the metric can be audited rather than only scored. |
| `rag.context_relevancy` | `context_relevancy` | **Same quantity.** |
| — | `answer_correctness` | **Not implemented as one metric,** on purpose. RAGAS blends factual overlap with semantic similarity into a single weighted score. AgentGate keeps `outcome.f1_token` and `outcome.semantic_similarity` separate: a blended number can hide which half moved, which is exactly the failure mode a gate must not have. |

## Judge metrics

| AgentGate | DeepEval equivalent | Relationship |
|---|---|---|
| `judge.instruction_following` | `GEval` with a custom criterion | **Same idea, more controlled.** Both ask a model to grade against a rubric. AgentGate additionally applies position-swap averaging, verbosity and markdown audits, judge/agent family independence (default-deny), a Cohen's κ calibration floor against human labels, and frozen-anchor drift detection. |
| `judge.coherence` | `GEval` coherence | Same idea. |
| `judge.groundedness` | `FaithfulnessMetric` | Same idea; AgentGate retains per-claim detail. |
| `judge.helpfulness` | `GEval` helpfulness | Same idea. |
| — | `HallucinationMetric` | Covered by `rag.faithfulness` plus `safety.*` tripwires rather than as one metric. |

**The substantive difference is not the prompt — it is what surrounds it.** A judge is an
instrument, and an instrument without a stated error is not a measurement. See
[Methodology](methodology.md) for the bias controls and the variance propagation.

## Trajectory metrics

Neither library covers these. Tool selection, ordering, argument correctness, step efficiency,
loop detection, and error recovery come from the τ-bench and AgentBench lines of work, and are
implemented here directly against the recorded trajectory.

---

## Running the numerical comparison yourself

If you have an API key and want the side-by-side that this project does not ship:

```bash
uv pip install ragas deepeval
export OPENAI_API_KEY=...           # or whichever provider those libraries are configured for

# 1. Record a run so the trajectories exist and can be replayed deterministically.
uv run agentgate run --suite suites/crm_ops --mode cache --store .agentgate/parity.duckdb

# 2. Export per-sample contexts, answers, and references in RAGAS's input shape.
uv run agentgate export --run-id <run-id> --suite suites/crm_ops \
  --store .agentgate/parity.duckdb --format ragas --target parity.jsonl
```

Then score `parity.jsonl` with both libraries and compare per-sample values, not suite means. Two
implementations can agree on the mean while disagreeing on every individual item, and only the
per-sample correlation tells you whether they are measuring the same thing.

Expect exact agreement on the deterministic metrics (`context_recall`, `context_precision` where
gold ids exist) and only correlation, not equality, on the LLM-graded ones — those depend on the
grading model, its temperature, and its prompt, all of which differ between implementations. A
report claiming exact agreement on an LLM-graded metric would be evidence of a mistake, not of
quality.
