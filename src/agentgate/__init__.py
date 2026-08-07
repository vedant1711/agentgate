"""AgentGate — a statistical regression gate for AI agents.

AgentGate runs an agent-under-test against versioned task suites, scores final answers *and*
full trajectories, quantifies uncertainty with paired statistics, and blocks a pull request
only when the agent has *statistically significantly* regressed beyond a declared margin.

See ``docs/methodology.md`` for the formulas and their citations.
"""

from __future__ import annotations

from importlib import metadata

try:  # pragma: no cover - trivial packaging branch
    __version__ = metadata.version("agentgate")
except metadata.PackageNotFoundError:  # pragma: no cover - editable/source checkouts
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
