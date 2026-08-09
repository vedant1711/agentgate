# Current results

Everything on this page is generated from `results/harness.json`, which is written by
`agentgate harness export` and committed on every recording session. Nothing here is typed by
hand, so it cannot drift from the evidence it describes.

**Read every value with its interval.** A point estimate on its own is the artifact this project
exists to argue against.


**1 recording(s)** across 1 suite(s) and 1 model(s), at 95% confidence.


!!! note "What these numbers are not"

    The τ² suite is a **single-turn adaptation** of τ²-bench, which is multi-turn with a simulated
    user. The tasks, the tool surface and the gold trajectories are τ²-bench's; the interaction
    protocol is ours. These are **not** τ²-bench leaderboard scores and must not be compared to
    them.

    Metrics absent from a table did not apply to that suite and were skipped, never scored zero.


## Llama 3.2 3B on `tau2_retail`

`ollama_chat/llama3.2:3b` · agent `tau2_retail_agent` · 111 tasks x K=3 = 333 units · 99% completed · recorded 2026-08-08


### Headline

| metric | value | 95% CI | n | method |
|---|---|---|---|---|
| `outcome.task_success` | 0.180 | [0.108, 0.252] | 111 | clt(task-rates) |
| `judge.instruction_following` | 0.619 | [0.559, 0.680] | 111 | clt |
| `judge.coherence` | 0.994 | [0.988, 1.000] | 111 | clt |
| `trajectory.recall` | 0.535 | [0.476, 0.595] | 111 | clt |
| `trajectory.precision` | 0.574 | [0.517, 0.631] | 111 | clt |
| `trajectory.argument_correctness` | 0.145 | [0.116, 0.173] | 111 | clt |
| `trajectory.error_recovery_rate` | 0.000 | [0.000, 0.000] | 105 | clt |
| `trajectory.step_efficiency` | 0.766 | [0.711, 0.821] | 111 | clt |

??? note "All other metrics"

    | metric | value | 95% CI | n | method |
    |---|---|---|---|---|
    | `efficiency.completion_tokens` | 253.853 | [234.532, 273.174] | 111 | clt |
    | `efficiency.est_cost_usd` | 0.000 | [0.000, 0.000] | 111 | clt |
    | `efficiency.latency_ms` | 13,260 | [12264.887, 14254.989] | 111 | clt |
    | `efficiency.llm_roundtrips` | 2.027 | [1.988, 2.066] | 111 | clt |
    | `efficiency.prompt_tokens` | 1,875 | [1836.916, 1913.102] | 111 | clt |
    | `efficiency.tool_calls_count` | 4.234 | [3.758, 4.711] | 111 | clt |
    | `efficiency.total_tokens` | 2,129 | [2077.419, 2180.305] | 111 | clt |
    | `outcome.abstained` | 0.000 | [0.000, 0.000] | 111 | clt(task-rates) |
    | `rag.answer_relevancy` | 0.267 | [0.243, 0.290] | 111 | clt |
    | `trajectory.any_order_match` | 0.225 | [0.147, 0.303] | 111 | clt(task-rates) |
    | `trajectory.exact_match` | 0.000 | [0.000, 0.000] | 111 | clt(task-rates) |
    | `trajectory.f1` | 0.458 | [0.421, 0.495] | 111 | clt |
    | `trajectory.in_order_match` | 0.189 | [0.116, 0.262] | 111 | clt(task-rates) |
    | `trajectory.lcs_ratio` | 0.501 | [0.445, 0.557] | 111 | clt |
    | `trajectory.loop_detected` | 0.000 | [0.000, 0.000] | 111 | clt(task-rates) |
    | `trajectory.redundant_call_rate` | 0.028 | [0.012, 0.044] | 111 | clt |
    | `trajectory.single_tool_use` | 0.324 | [0.237, 0.412] | 111 | clt(task-rates) |
