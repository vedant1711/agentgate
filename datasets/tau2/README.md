# tau2-bench data (vendored)

`retail-db.json` and `retail-tasks.json` come from
[sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench), **MIT licensed**.

They are vendored rather than downloaded at runtime so the suite is reproducible and CI never
depends on a third-party repository staying up. Re-fetch with:

```
uv run python scripts/build_tau2_suite.py --download
```

Only the **retail** domain is vendored, because retail is the domain AgentGate implements
(`agentgate/agents/tau2_retail.py`). Shipping airline data we cannot execute would be dead weight.

## What we changed

tau2-bench is multi-turn: the agent converses with a simulated user who withholds information
until asked. AgentGate is single-turn. Each task's scenario is flattened into one instruction
carrying `known_info`; `unknown_info` is withheld so the agent must still discover it with tools.

**Scores from `suites/tau2_retail` are not tau2-bench leaderboard scores.** The tasks, tool
surface, and gold trajectories are theirs; the interaction protocol is ours.
