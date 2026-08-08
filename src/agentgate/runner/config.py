"""Run configuration and identity.

The run id is *derived*, not random: it is a function of the suite, agent, K, seed, mode, and
active faults. That makes resume work without a state file, makes re-running the same config a
no-op instead of a duplicate, and makes two runs that should be comparable obviously so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agentgate.faults.config import FaultConfig
from agentgate.providers.client import DEFAULT_CACHE_PATH
from agentgate.schemas.common import ProviderMode, stable_hash
from agentgate.schemas.results import BudgetSpec

DEFAULT_BASE_SEED = 20260101
DEFAULT_RUN_ROOT = Path(".agentgate/runs")


@dataclass(slots=True)
class RunConfig:
    """Everything the runner needs to execute one system against one suite."""

    suite_path: Path
    system: str = "baseline"
    agent: str | None = None
    """Overrides the suite's declared agent."""
    k: int | None = None
    """Overrides the suite's ``default_k``."""
    mode: ProviderMode = ProviderMode.MOCK
    base_seed: int = DEFAULT_BASE_SEED
    concurrency: int = 4
    budget: BudgetSpec = field(default_factory=BudgetSpec)
    faults: FaultConfig = field(default_factory=FaultConfig)
    model: str = "mock/agent"
    run_root: Path = DEFAULT_RUN_ROOT
    store_path: Path | None = None
    cache_path: Path = Path(DEFAULT_CACHE_PATH)
    resume: bool = True
    run_id: str | None = None
    """Explicit id; otherwise derived from the configuration."""

    def identity(
        self, *, suite_name: str, suite_hash: str, agent: str, k: int
    ) -> dict[str, object]:
        """Return the fields the derived run id is computed over."""
        return {
            "suite": suite_name,
            "suite_hash": suite_hash,
            "agent": agent,
            "k": k,
            "base_seed": self.base_seed,
            "mode": self.mode.value,
            "model": self.model,
            "faults": self.faults.active(),
        }

    def derive_run_id(self, *, suite_name: str, suite_hash: str, agent: str, k: int) -> str:
        """Build a stable, human-legible run id.

        The ``system`` label is part of the id but not the hash, so a no-op comparison (baseline
        vs candidate with identical settings) still yields two distinct, resumable runs.
        """
        if self.run_id:
            return self.run_id
        digest = stable_hash(
            self.identity(suite_name=suite_name, suite_hash=suite_hash, agent=agent, k=k), length=10
        )
        return f"{suite_name}-{self.system}-{digest}"

    def run_dir(self, run_id: str) -> Path:
        """Directory holding this run's JSONL trajectories and artifacts."""
        return self.run_root / run_id
