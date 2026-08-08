# Methodology

Every number AgentGate prints, and why it is that number rather than another one.

---

## 1. The question a gate should ask

A conventional agent CI check asks:

```python
if new_score < old_score - 0.03:
    fail_the_build()
```

On a 30-case golden set this rule is a coin flip. If task success is around 0.85, the standard
error of the mean is roughly `sqrt(0.85 × 0.15 / 30) ≈ 0.065`. A 3-point move is less than half
an SE — it happens constantly on an unchanged agent, and it fails to fire on real regressions
smaller than the noise floor. The rule has no error rate because it never computes one.

AgentGate asks a different question, and — critically — can give **three** answers:

| Verdict | Meaning | Formal statement |
|---|---|---|
| `REGRESSION` | The candidate is worse by more than the margin, and the evidence supports saying so | Reject `H₀: Δ ≥ −δ` at the FDR-adjusted level |
| `PASS` | The candidate is *proven* fine, not merely unproven bad | Reject `H₀: Δ ≤ −δ` at α |
| `UNDERPOWERED` | Neither rejects, and the suite could not have detected δ anyway | Neither rejection, and achieved power < target |

The third answer is the point. A naive gate is silent exactly when it knows least, and its
silence is indistinguishable from a pass.

---

## 2. The analysis unit

**Tasks, never repetitions.** Each task runs `K` times; the `K` scores are averaged into one
task-level score, and every suite-level statistic runs over those. Repetitions of the same task
are not independent draws from the task distribution; treating them as such would shrink every
interval by a factor of `√K` for nothing.

**Clusters, when the suite declares them.** `suites/crm_ops` contains five paraphrases of each
scenario. Those five are one fact, not five. Both the tests and the intervals therefore operate
on **per-cluster mean differences**. Implemented in
[`stats/compare.py`](../src/agentgate/stats/compare.py).

---

## 3. Intervals

| Regime | Procedure | Why |
|---|---|---|
| Continuous / proportion means | CLT: `x̄ ± 1.96·s/√n` | Standard, and adequate at n ≥ 30 |
| Binary, K = 1 | Wilson score interval (Wilson, 1927) | The normal approximation has genuinely bad coverage at small n and extreme p — exactly the regime of a 30–200 task suite where a good agent scores 0.9+ |
| Clustered designs | Cluster-robust sandwich with CR1 correction | Naive SEs understate uncertainty by >2× in our own simulation |
| Bounded, skewed statistics | BCa bootstrap, 10,000 resamples | pass^k and flake rate live near their bounds where a normal interval misbehaves |

**Cluster-robust variance of a mean:**

```
V = G/(G−1) · Σ_g ( Σ_i (y_gi − ȳ) )² / n²
```

reported alongside the naive SE with the inflation ratio, so a reader sees the difference rather
than being told about it.

**BCa** corrects two things the percentile interval gets wrong: bias (`z₀`, from the share of
bootstrap replicates below the observed statistic) and skew (acceleration `a`, from the
jackknife). Resampling is done **at cluster level** when clusters exist — resampling individual
tasks would destroy the dependence the clustering represents.

*Note.* Proportion intervals are clipped to [0, 1]. An interval on a success rate that reaches
−0.05 is arithmetic that was not thought about.

---

## 4. Pairing

Runs to be compared must use identical task instances, identical `K`, and identical seeds. The
runner refuses mismatched comparisons rather than warning about them, because a paired test
across different task instances is not a paired test at all.

Pairing is not bookkeeping — it is the single highest-leverage decision in the design:

```
Var(d) = Var(x) + Var(y) − 2·r·SD(x)·SD(y)
```

At `r = 0.5` with equal variances the paired variance is **half** the unpaired variance: the same
precision for half the sample, for free. Two versions of an agent correlate 0.3–0.7 on per-task
scores, so this is not a theoretical gain. The report prints the realised `r`, and the
Monte-Carlo suite reproduces the relationship on synthetic data where the correlation is *set*
rather than observed.

---

## 5. Tests

| dtype | Primary | Fallback | Robustness check |
|---|---|---|---|
| binary | Paired t on task-level rates, shifted by δ | Permutation | Exact McNemar at δ = 0, reported alongside |
| proportion / continuous | Paired t, shifted by δ | Wilcoxon (δ = 0) or permutation (δ ≠ 0) | Permutation, 20,000 sign flips |

**The margin is a shift.** Testing `dᵢ + δ` against zero *is* testing `mean(d) > −δ`, which is
the hypothesis the gate cares about.

**Test selection** runs Shapiro–Wilk on the differences and logs the choice, so no p-value in a
report is of unknown provenance. Two refinements the naive rule gets wrong:

1. **Heavily tied differences.** When most tasks are unchanged, shifting by δ turns every tied
   zero into a small positive, and those tiny ranks outvote a handful of large genuine negatives.
   Above a 30% tie fraction, the sign-flip permutation test is used instead.
2. **Any margin at all.** Wilcoxon tests a *median* shift; the margin is defined on the **mean**.
   On skewed differences the two genuinely disagree — a candidate comfortably inside the margin
   can have a median outside it. Whenever `δ ≠ 0`, the non-normal fallback is the permutation
   test, which is distribution-free *and* about the mean.

**McNemar** conditions on the `b + c` discordant pairs: a task both systems passed carries no
information about which is better. One-sided `p = P(X ≤ c)` for `X ~ Binom(b+c, ½)`.

**Permutation** p-values use `(1 + hits)/(1 + B)`, which keeps them strictly positive — a p-value
of exactly 0 would claim more evidence than `B` resamples can supply.

---

## 6. Multiplicity

Twelve metrics at raw α = 0.05 gives a **46% chance of at least one false alarm** per comparison.
That is how a statistical gate becomes a gate everyone learns to ignore.

Benjamini–Hochberg controls the false discovery rate — the expected share of *rejections* that
are false — which is the right error rate for a gate. Bonferroni controls the probability of any
false rejection at all, which is stricter than needed and hides real regressions. BH is uniformly
less conservative: every metric Bonferroni rejects, BH rejects too (asserted as a property test).

Adjusted p-values use the step-up transform `min_{j≥i} (m/j)·p_(j)`, and **rejections are derived
from the adjusted values**, not from a separately computed threshold. The two are equivalent in
exact arithmetic but can differ by one ULP — and a report that prints "adjusted p = 0.05, not
rejected" at q = 0.05 is indefensible however correct its floating point.

Raw and adjusted p-values both appear in every report.

---

## 7. Power and the minimum detectable effect

Every gate report prints:

> *This suite can detect a change of X.XX in `outcome.task_success` with 80% power; your
> configured margin is δ = Y.YY.*

Power uses the **non-central t** distribution, not the normal approximation, which at n = 30
overstates power by several points — and overstating power is the specific failure this module
exists to prevent. The MDE is solved numerically against the same power curve the test actually
uses.

`agentgate plan --target-mde 0.03` inverts it. On the 8-task smoke suite:

```
target MDE       0.03
assumed sigma_d  0.15
tasks required   156

smoke has 8 tasks: it can detect 0.147 at 80% power, and has 13% power for your 0.03 target.
underpowered: write about 148 more task(s), or widen the margin to 0.1467
```

---

## 8. Reliability: pass@k and pass^k

τ-bench's observation: a GPT-4o agent averaging above 60% success drops below 25% at pass^8.
Average success and *reliability* are different quantities, and only the first is visible in a
single-run eval.

Both estimators are the unbiased combinatorial forms, not plug-in rates:

```
pass@k  = 1 − C(K−c, k) / C(K, k)      probability at least one of k draws succeeds
pass^k  =     C(c, k)   / C(K, k)      probability all k draws succeed
```

The naive `(c/K)^k` is biased, and the bias is largest exactly where it matters, near p = 1. The
Monte-Carlo suite verifies mean-unbiasedness directly and shows the naive form measurably worse.

`k > K` raises rather than extrapolating: estimating pass^8 from four repetitions would be an
invention, not an estimate.

---

## 9. The judge as an instrument

Judge-backed metrics carry the judge's own measurement error into their standard error, via the
law of total variance:

```
SE² = Var(task means)/n + mean(judge variance) / (J · K · n)
```

The judge's per-draw variance is damped by J draws, K repetitions, and n tasks — but it does not
vanish, and ignoring it reports an interval narrower than the evidence supports.

Bias is **measured**, not assumed away:

| Control | Mechanism | Verification |
|---|---|---|
| Position bias | Both slot orders, averaged | A synthetic judge with an anchored 30% slot-A preference produces a measured 30% flip rate |
| Verbosity bias | Spearman(length, score); flagged above 0.4 | A verbosity-loving judge trips the audit; an unbiased one does not |
| Format bias | Same, on markdown density | — |
| Self-preference | Judge/agent family independence, default-deny | Override attaches a SELF-JUDGING warning to every report |
| Calibration | Cohen's κ and Spearman ρ vs human labels | κ checked against hand-computed examples, including the 0/0 case |
| Drift | Frozen 30-item anchor set, 95% band | Banners "scores not comparable to history" |

A criterion whose judge scores κ < 0.6 may be **reported** but may not **gate**. A gate on an
unvalidated instrument measures the instrument.

---

## 10. Validation

`tests/simulation/` runs ≥2,000 simulated experiments per case against synthetic agents whose
truth is set by the experimenter. Results:

| Property | Requirement | Observed |
|---|---|---|
| Type-I error, identical systems | 0.03–0.07 at nominal 0.05 | inside band |
| CI coverage (CLT, Wilson, paired) | 0.93–0.97 | inside band |
| Empirical power vs predicted | within ±5 pts | inside band |
| pass^k unbiasedness | converges to `p^k` | inside Monte-Carlo error |
| Naive SEs under clustering | should under-cover | 76% coverage; cluster-robust repairs it |
| Pairing benefit | grows with correlation | monotone across r = 0, 0.3, 0.5, 0.7 |

Plus a **false-positive control**: 20 repeated no-change comparisons at different seeds yield at
most 2 regressions.

---

## References

1. Yao et al. — *τ-bench* (arXiv:2406.12045). pass^k; goal-state checking.
2. Miller (Anthropic) — *Adding Error Bars to Evals* (arXiv:2411.00640). CLT SEs, clustered SEs, paired differences, resampling, power.
3. Zheng et al. — *Judging LLM-as-a-Judge* (arXiv:2306.05685). Position, verbosity, self-enhancement bias.
4. Wang et al. — *LLMs are not Fair Evaluators* (arXiv:2305.17926). Position-bias magnitude; swap-and-average.
5. Panickssery et al. — self-preference bias (arXiv:2410.21819).
6. Liu et al. — *G-Eval* (arXiv:2303.16634). Rubric + CoT judge design.
7. Google Cloud — *Evaluate Gen AI agents*. Trajectory match definitions.
8. *TRACE* (arXiv:2510.02837). Efficiency, evidence-grounded reasoning.
9. Es et al. — *RAGAS* (arXiv:2309.15217).
10. Chen et al. — *Evaluating LLMs Trained on Code* (arXiv:2107.03374). Unbiased pass@k.
11. Benjamini & Hochberg (1995). FDR control.
12. Wilson (1927); McNemar (1947); Tango (1998).
