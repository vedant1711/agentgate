"""The K-repetition scheduler (L2).

Executes ``N tasks x K repetitions`` of one system against one suite, with deterministic
per-unit seeds, bounded concurrency, resume-from-partial, and a clean budget halt.

Two properties matter more than throughput:

* **Determinism.** Every unit's seed is ``derive_seed(base_seed, task_id, rep)``, so results do
  not depend on scheduling order, concurrency level, or which units a resumed run re-executed.
* **Durability.** Trajectories are flushed to JSONL as they complete, so an interrupted run
  loses nothing and resumes to a byte-identical result (Phase 3 acceptance).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from agentgate import __version__
from agentgate.agents.protocol import AgentConfig, AgentUnderTest
from agentgate.agents.registry import build_agent
from agentgate.errors import BudgetExceededError, ConfigError
from agentgate.provenance import git_dirty, git_sha, host_info, library_versions
from agentgate.providers.catalog import estimate_cost
from agentgate.providers.client import LLMClient
from agentgate.runner.config import RunConfig
from agentgate.runner.loader import load_suite, validate_suite
from agentgate.schemas.common import sha256_hex
from agentgate.schemas.results import ModelRef, RunManifest, RunSummary
from agentgate.schemas.task import SuiteRef, SuiteSpec, TaskSpec
from agentgate.schemas.trajectory import RunStatus, Trajectory
from agentgate.seeds import derive_seed
from agentgate.storage.duckdb_store import RunStore
from agentgate.storage.jsonl import TrajectoryWriter, load_trajectories

ProgressHook = Callable[[int, int, str], None]
"""``(completed, total, unit_label)`` — called after each unit finishes."""


@dataclass(slots=True)
class RunResult:
    """Everything a completed run produced."""

    manifest: RunManifest
    trajectories: list[Trajectory]
    summary: RunSummary
    warnings: list[str] = field(default_factory=list)
    resumed: int = 0
    """Units served from a previous partial run rather than re-executed."""

    @property
    def run_id(self) -> str:
        """The run's identifier."""
        return self.manifest.run_id


@dataclass(frozen=True, slots=True)
class _Unit:
    """One (task, repetition) execution unit."""

    index: int
    task: TaskSpec
    rep: int
    seed: int

    @property
    def key(self) -> tuple[str, int]:
        """Resume key."""
        return (self.task.id, self.rep)

    @property
    def label(self) -> str:
        """Human-readable unit name for progress output."""
        return f"{self.task.id}#{self.rep}"


class Runner:
    """Executes a suite for one system.

    Args:
        config: Run configuration.
        suite: Pre-loaded suite; loaded from ``config.suite_path`` when omitted.
        agent: Pre-built agent; built from the registry when omitted.
        store: DuckDB store; opened at ``config.store_path`` when omitted and a path is set.
    """

    def __init__(
        self,
        config: RunConfig,
        *,
        suite: SuiteSpec | None = None,
        agent: AgentUnderTest | None = None,
        store: RunStore | None = None,
    ) -> None:
        self.config = config
        self.suite = suite if suite is not None else load_suite(config.suite_path)
        self.agent_name = config.agent or self.suite.agent
        self.k = config.k or self.suite.default_k
        self._agent = agent
        self._store = store
        self._owns_store = store is None

    # -- identity ----------------------------------------------------------

    @property
    def run_id(self) -> str:
        """The derived (or explicitly configured) run id."""
        return self.config.derive_run_id(
            suite_name=self.suite.name,
            suite_hash=self.suite.content_digest(),
            agent=self.agent_name,
            k=self.k,
        )

    def units(self) -> list[_Unit]:
        """Enumerate every (task, repetition) unit with its deterministic seed."""
        units: list[_Unit] = []
        for task in self.suite.tasks:
            reps = task.k or self.k
            for rep in range(reps):
                units.append(
                    _Unit(
                        index=len(units),
                        task=task,
                        rep=rep,
                        seed=derive_seed(self.config.base_seed, task.id, rep),
                    )
                )
        return units

    def build_manifest(self, agent: AgentUnderTest) -> RunManifest:
        """Capture everything needed to reproduce this run (A3.4)."""
        prompt_hashes: dict[str, str] = {}
        system_prompt = getattr(agent, "system_prompt", None)
        if callable(system_prompt):
            prompt_hashes["system"] = sha256_hex(system_prompt(self.suite.tasks[0]))[:16]
        return RunManifest(
            run_id=self.run_id,
            created_at=datetime.now(UTC),
            agentgate_version=__version__,
            git_sha=git_sha(),
            git_dirty=git_dirty(),
            suite=SuiteRef.from_suite(self.suite),
            system=self.config.system,
            agent=self.agent_name,
            k=self.k,
            base_seed=self.config.base_seed,
            mode=self.config.mode,
            models=[
                ModelRef(role="agent", model_id=self.config.model, provider=self.config.mode.value)
            ],
            prompt_hashes=prompt_hashes,
            library_versions=library_versions(),
            faults=self.config.faults.active(),
            budget=self.config.budget,
            host=host_info(),
        )

    # -- execution ---------------------------------------------------------

    async def run(self, *, progress: ProgressHook | None = None) -> RunResult:
        """Execute the suite and return its trajectories.

        Args:
            progress: Optional callback invoked after each unit completes.

        Returns:
            The run's manifest, trajectories (sorted by task then repetition), and summary.

        Raises:
            ConfigError: When the configured agent is unknown.
        """
        started = time.perf_counter()
        agent: AgentUnderTest = self._agent or build_agent(
            self.agent_name,
            mode=self.config.mode,
            faults=self.config.faults,
            config=AgentConfig(model=self.config.model, system=self.config.system),
            cache_path=self.config.cache_path,
            budget=self.config.budget,
            system=self.config.system,
        )
        manifest = self.build_manifest(agent)
        warnings = validate_suite(self.suite)

        run_dir = self.config.run_dir(manifest.run_id)
        jsonl_path = run_dir / "trajectories.jsonl"
        done: dict[tuple[str, int], Trajectory] = {}
        if self.config.resume and jsonl_path.exists():
            done = {
                (trajectory.task_id, trajectory.rep): trajectory
                for trajectory in load_trajectories(jsonl_path)
            }
            if done:
                warnings.append(f"resumed {len(done)} unit(s) from {jsonl_path}")

        units = self.units()
        pending = [unit for unit in units if unit.key not in done]
        results: dict[tuple[str, int], Trajectory] = dict(done)
        halted: str | None = None
        completed = len(done)

        writer = TrajectoryWriter(jsonl_path, append=bool(done))
        semaphore = asyncio.Semaphore(max(1, self.config.concurrency))
        lock = asyncio.Lock()
        stop = asyncio.Event()

        async def execute(unit: _Unit) -> None:
            nonlocal halted, completed
            if stop.is_set():
                return
            async with semaphore:
                if stop.is_set():
                    return
                try:
                    trajectory = await agent.run(unit.task, unit.seed)
                except BudgetExceededError as exc:
                    # A budget stop is a harness decision, not an agent failure: halt cleanly
                    # and keep everything already produced.
                    stop.set()
                    halted = str(exc)
                    return
                trajectory.rep = unit.rep
                trajectory.system = self.config.system
                trajectory.est_cost_usd = estimate_cost(self.config.model, trajectory.usage)
                async with lock:
                    results[unit.key] = trajectory
                    writer.write(trajectory)
                    completed += 1
                    if progress is not None:
                        progress(completed, len(units), unit.label)

        try:
            await asyncio.gather(*(execute(unit) for unit in pending))
        finally:
            writer.close()

        ordered: list[Trajectory] = []
        for unit in units:
            trajectory = results.get(unit.key)
            if trajectory is None:
                trajectory = _unfinished(unit, self.config.system, self.agent_name, halted)
            ordered.append(trajectory)

        if halted is not None:
            warnings.append(f"run halted early: {halted}")

        summary = self._summarise(manifest, ordered, agent, time.perf_counter() - started)
        self._persist(manifest, ordered)
        return RunResult(
            manifest=manifest,
            trajectories=ordered,
            summary=summary,
            warnings=warnings,
            resumed=len(done),
        )

    def _summarise(
        self,
        manifest: RunManifest,
        trajectories: list[Trajectory],
        agent: AgentUnderTest,
        wall_seconds: float,
    ) -> RunSummary:
        counts: dict[str, int] = {}
        for trajectory in trajectories:
            counts[trajectory.status.value] = counts.get(trajectory.status.value, 0) + 1
        client: LLMClient | None = getattr(agent, "client", None)
        stats = client.stats if client is not None else None
        return RunSummary(
            run_id=manifest.run_id,
            manifest=manifest,
            n_tasks=len(self.suite.tasks),
            n_samples=len(trajectories),
            status_counts=counts,
            total_tokens=sum(t.usage.total_tokens for t in trajectories),
            total_cost_usd=sum(t.est_cost_usd for t in trajectories),
            wall_seconds=wall_seconds,
            cache_hits=stats.cache_hits if stats else 0,
            cache_misses=stats.cache_misses if stats else 0,
        )

    def _persist(self, manifest: RunManifest, trajectories: list[Trajectory]) -> None:
        if self.config.store_path is None and self._store is None:
            return
        store = self._store or RunStore(self.config.store_path or ":memory:")
        try:
            store.save_run(manifest)
            store.save_samples(
                manifest.run_id,
                [t for t in trajectories if t.status is not RunStatus.BUDGET_EXHAUSTED],
                {task.id: task.cluster_id for task in self.suite.tasks},
            )
        finally:
            if self._owns_store and self._store is None:
                store.close()


def _unfinished(unit: _Unit, system: str, agent: str, reason: str | None) -> Trajectory:
    """Placeholder for a unit the budget halt prevented from running."""
    return Trajectory(
        task_id=unit.task.id,
        rep=unit.rep,
        seed=unit.seed,
        system=system,
        agent=agent,
        status=RunStatus.BUDGET_EXHAUSTED,
        error=reason or "not executed",
    )


async def run_suite(config: RunConfig, *, progress: ProgressHook | None = None) -> RunResult:
    """Convenience wrapper: build a runner from ``config`` and execute it.

    Args:
        config: Run configuration.
        progress: Optional progress callback.

    Returns:
        The run result.

    Raises:
        ConfigError: When the suite or agent cannot be resolved.
    """
    runner = Runner(config)
    if runner.agent_name not in _known_agents():
        msg = f"unknown agent {runner.agent_name!r} declared by suite {runner.suite.name!r}"
        raise ConfigError(msg)
    return await runner.run(progress=progress)


def _known_agents() -> set[str]:
    from agentgate.agents.registry import AGENT_CLASSES

    return set(AGENT_CLASSES)
