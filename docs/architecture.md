# Architecture

Seven layers, each with one job. The value of the split is that **every layer can be tested
without the ones above it**, and the whole thing can run with no network at all.

---

## The whole system

```mermaid
flowchart TB
    subgraph L1["1 · Suite"]
        A[Tasks + gold trajectories<br/>YAML, content-hashed]
    end
    subgraph L2["2 · Runner"]
        B[Executes task x K<br/>derived seeds, resumable]
    end
    subgraph L3["3 · Agent + sandbox"]
        C[ReAct loop]
        D[(Mock company<br/>10 tools, SQLite)]
        C <--> D
    end
    subgraph L4["4 · Provider"]
        E{{live · cache · replay · mock}}
    end
    subgraph L5["5 · Metrics"]
        F[42 metrics, per sample]
        G[LLM judge<br/>bias-controlled]
        F --> G
    end
    subgraph L6["6 · Statistics"]
        H[Paired · clustered · BH-adjusted]
    end
    subgraph L7["7 · Gate"]
        I{Three-way verdict}
    end

    A --> B --> C
    C <--> E
    C --> J[Trajectories<br/>every step recorded]
    J --> F
    G --> K[(DuckDB<br/>per-sample scores)]
    F --> K
    K --> H --> I
    I --> M[PR comment + HTML report]

    style L4 fill:#e0f2f1,stroke:#00695c
    style L6 fill:#e8eaf6,stroke:#283593
    style L7 fill:#fff3e0,stroke:#ef6c00
```

**Why the trajectory is the hinge.** Everything above it produces one; everything below it only
reads one. That means metrics, statistics, and the gate can all be tested against recorded
trajectories with no model, no network, and no agent — which is why the test suite runs in about
a minute and why CI never spends a token.

---

## The two speeds

```mermaid
flowchart LR
    subgraph slow["Recording — hours, in the background"]
        direction TB
        S1[Call the model] --> S2[Record the response]
        S2 --> S3[(Cache<br/>SQLite)]
    end
    subgraph fast["Deciding — seconds, in CI"]
        direction TB
        F1[Replay from cache] --> F2[Score] --> F3[Rule]
    end
    S3 -.->|"same prompt, same answer,<br/>forever"| F1
    style slow fill:#fff3e0,stroke:#ef6c00
    style fast fill:#e0f2f1,stroke:#00695c
```

A 70-task suite at K=4 is roughly 1,120 model calls. On a local model that is **45 minutes to a
full day**; on a free tier it is hours of rate-limited waiting. Deciding, once the responses
exist, takes seconds.

That gap is the reason for the provider layer's four modes:

| Mode | Calls the model? | Used for |
|---|---|---|
| `live` | Always | Never in CI. Ad-hoc exploration only. |
| `cache` | Only on a miss | Recording. Write-through, so the next run is free. |
| `replay` | Never | **The CI default.** A miss is an error, not a silent live call. |
| `mock` | Never | Offline development against a deterministic stand-in. |

`replay` failing loudly on a cache miss is the load-bearing detail. A mode that quietly fell back
to `live` would make CI non-deterministic and would spend money without telling anyone.

---

## How a verdict is reached

```mermaid
flowchart TB
    A[Baseline vs candidate<br/>same tasks, same seeds] --> B[Per-task differences]
    B --> C{Any safety<br/>tripwire fired?}
    C -->|yes| D[SAFETY_FAIL<br/>skip the statistics entirely]
    C -->|no| E[Group into clusters<br/>5 rewordings = 1 fact]
    E --> F[One-sided test<br/>against margin δ]
    F --> G[Benjamini–Hochberg<br/>across the metric family]
    G --> H{Rejected<br/>beyond δ?}
    H -->|yes| I[REGRESSION]
    H -->|no| J{Non-inferiority<br/>established?}
    J -->|yes| K[PASS]
    J -->|no| L[UNDERPOWERED]

    style D fill:#ede7f6,stroke:#4527a0
    style I fill:#ffebee,stroke:#c62828
    style K fill:#e8f5e9,stroke:#2e7d32
    style L fill:#fff8e1,stroke:#ef6c00
```

Two things in that diagram carry most of the weight.

**Safety bypasses everything.** A prompt-injection compliance, a PII leak, a forbidden tool call
— these are not averaged. They short-circuit to a failure before any statistic is computed,
because "the average was fine" is not a defence for a security breach.

**`PASS` is a positive finding, not an absence.** It means non-inferiority was *established* —
the suite was powerful enough to have detected a regression of the size you declared you cared
about, and did not. When the suite is not that powerful, the answer is `UNDERPOWERED`, and
reading that as a pass is exactly the mistake this design prevents.

---

## Why clustering changes the answer

A suite usually contains several rewordings of the same underlying scenario. Treating them as
independent inflates the sample size, which shrinks the error bars, which makes the gate fire on
noise.

```mermaid
flowchart LR
    subgraph naive["Counted naively — 10 independent facts"]
        direction TB
        N["refund small · refund small (reworded)<br/>refund small (reworded) · refund small (reworded)<br/>… x10"]
    end
    subgraph real["Counted honestly — 2 independent facts"]
        direction TB
        R["cluster: refund_small<br/>cluster: escalation"]
    end
    naive -->|"error bars too narrow<br/>→ false alarms"| X[Wrong]
    real -->|"error bars honest"| Y[Right]
    style X fill:#ffebee,stroke:#c62828
    style Y fill:#e8f5e9,stroke:#2e7d32
```

Both the tests and the intervals operate on **per-cluster mean differences**, with CR1
cluster-robust standard errors. The reports show the inflation factor explicitly, so you can see
what clustering cost you rather than being told to trust it.

---

## The judge is an instrument, not an oracle

When a metric comes from an LLM grading text, that grade has error. Ignoring it produces
confident numbers with no basis.

```mermaid
flowchart TB
    A[Answer to grade] --> B[Rubric-decomposed<br/>criteria]
    B --> C[Position-swap<br/>averaging]
    C --> D[J independent draws]
    D --> E[Per-item variance]
    E --> F[Law of total variance<br/>folded into the metric SE]

    B -.-> G[Verbosity + markdown<br/>Spearman audits]
    B -.-> H[Judge/agent family<br/>independence · default-deny]
    B -.-> I["Cohen's κ floor<br/>against human labels"]
    B -.-> J[Frozen anchor set<br/>drift detection]

    style F fill:#e8eaf6,stroke:#283593
```

Each control answers a specific, documented way LLM judges go wrong: they prefer whichever answer
came first, they reward length and formatting regardless of content, they favour their own
family's outputs, and they drift as the provider updates the model underneath you.

---

## Reproducibility

```mermaid
flowchart LR
    A[Suite content hash] --> Z[config_hash]
    B[Agent + K + seed] --> Z
    C[Provider mode] --> Z
    D[Model + version pins] --> Z
    E[Prompt hashes] --> Z
    F[Library versions] --> Z
    G[Active fault knobs] --> Z
    Z --> R[Same inputs<br/>same verdict<br/>byte for byte]
    H[Host / OS / wall-clock] -.->|deliberately excluded| Z
    style R fill:#e0f2f1,stroke:#00695c
```

Host details are excluded from the hash **on purpose**: two runs on different machines with the
same inputs and the same replay cache must hash identically, or the hash is measuring the wrong
thing.

---

## Where the code lives

| Layer | Module | What it owns |
|---|---|---|
| Schemas | `agentgate/schemas/` | Every contract, pydantic v2, exported as JSON Schema |
| Provider | `agentgate/providers/` | Four modes, cache, rate limits, retries, budgets |
| Agents | `agentgate/agents/` | ReAct loops, sandbox, τ² adapter, fault injection |
| Runner | `agentgate/runner/` | Scheduling, seeds, resume, persistence |
| Metrics | `agentgate/metrics/` | 42 metrics, checkers, requirement gating |
| Judge | `agentgate/judge/` | Rubrics, bias audits, calibration, drift |
| Statistics | `agentgate/stats/` | Intervals, paired tests, multiplicity, power |
| Gate | `agentgate/gate/` | The three-way rule, policy, safety tripwires |
| Harness | `agentgate/harness/` | Ledger, scheduler, leaderboard, trends |
| Report | `agentgate/report/` | PR comment, HTML report, demo page |
