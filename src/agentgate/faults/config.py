"""Fault-injection knobs (F2).

A gate is only convincing when you can watch it catch a *real* regression, so AgentGate ships
the regressions too. Each knob reproduces a failure class that actually happens to agent
systems in production — a prompt someone "simplified", a tool renamed in a refactor, a
cost-driven model downgrade — rather than an artificial score perturbation.

Knobs are read from the environment so a CI job can flip one without touching code, which is
exactly how the example failing PR (K4) works.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from pydantic import Field

from agentgate.schemas.common import FrozenModel

ENV_PREFIX = "FAULT_"


def _as_bool(value: str | None) -> bool:
    """Interpret an environment string as a flag."""
    return value is not None and value.strip().lower() not in ("", "0", "false", "no", "off")


def _as_float(value: str | None, default: float) -> float:
    """Interpret an environment string as a float, falling back on garbage."""
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


class FaultConfig(FrozenModel):
    """Which regressions are active for this run.

    An all-default instance is the healthy baseline; the run manifest records
    :meth:`active` so a report can never misattribute a regression to the wrong cause.
    """

    prompt_degrade: bool = Field(
        default=False, description="Someone 'simplified' the system prompt and dropped policy."
    )
    drop_tool: str | None = Field(
        default=None, description="A tool silently removed or renamed in a refactor."
    )
    truncate_context: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Multiplier on retrieval top-k / context budget. 0.5 halves it.",
    )
    model_swap: str | None = Field(
        default=None, description="Cost-driven downgrade to a weaker model."
    )
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Sampling-config drift."
    )
    tool_latency_ms: float = Field(
        default=0.0, ge=0.0, description="Flaky downstream dependency: added per-call latency."
    )
    tool_error_rate: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Flaky downstream dependency: per-call failure."
    )
    injection_vuln: bool = Field(
        default=False, description="Removes the agent's injection-hardening instructions."
    )
    verbosity: bool = Field(
        default=False, description="Answers 3x longer with no added content (judge bias probe)."
    )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> FaultConfig:
        """Build a config from ``FAULT_*`` environment variables.

        Args:
            env: Mapping to read; defaults to ``os.environ``.

        Returns:
            The parsed configuration. Unknown or malformed values fall back to healthy defaults
            rather than raising, because a typo in CI should not look like a passing gate.
        """
        source = os.environ if env is None else env
        drop_tool = source.get(f"{ENV_PREFIX}DROP_TOOL") or None
        model_swap = source.get(f"{ENV_PREFIX}MODEL_SWAP") or None
        raw_temperature = source.get(f"{ENV_PREFIX}TEMPERATURE")
        return cls(
            prompt_degrade=_as_bool(source.get(f"{ENV_PREFIX}PROMPT_DEGRADE")),
            drop_tool=drop_tool,
            truncate_context=min(
                1.0, max(0.01, _as_float(source.get(f"{ENV_PREFIX}TRUNCATE_CONTEXT"), 1.0))
            ),
            model_swap=model_swap,
            temperature=None if raw_temperature is None else _as_float(raw_temperature, 0.0),
            tool_latency_ms=max(0.0, _as_float(source.get(f"{ENV_PREFIX}TOOL_LATENCY_MS"), 0.0)),
            tool_error_rate=min(
                1.0, max(0.0, _as_float(source.get(f"{ENV_PREFIX}TOOL_ERROR_RATE"), 0.0))
            ),
            injection_vuln=_as_bool(source.get(f"{ENV_PREFIX}INJECTION_VULN")),
            verbosity=_as_bool(source.get(f"{ENV_PREFIX}VERBOSITY")),
        )

    def active(self) -> list[str]:
        """Return the knob strings to record in the run manifest, sorted for determinism."""
        knobs: list[str] = []
        if self.prompt_degrade:
            knobs.append(f"{ENV_PREFIX}PROMPT_DEGRADE=1")
        if self.drop_tool:
            knobs.append(f"{ENV_PREFIX}DROP_TOOL={self.drop_tool}")
        if self.truncate_context < 1.0:
            knobs.append(f"{ENV_PREFIX}TRUNCATE_CONTEXT={self.truncate_context:g}")
        if self.model_swap:
            knobs.append(f"{ENV_PREFIX}MODEL_SWAP={self.model_swap}")
        if self.temperature is not None:
            knobs.append(f"{ENV_PREFIX}TEMPERATURE={self.temperature:g}")
        if self.tool_latency_ms > 0:
            knobs.append(f"{ENV_PREFIX}TOOL_LATENCY_MS={self.tool_latency_ms:g}")
        if self.tool_error_rate > 0:
            knobs.append(f"{ENV_PREFIX}TOOL_ERROR_RATE={self.tool_error_rate:g}")
        if self.injection_vuln:
            knobs.append(f"{ENV_PREFIX}INJECTION_VULN=1")
        if self.verbosity:
            knobs.append(f"{ENV_PREFIX}VERBOSITY=1")
        return sorted(knobs)

    @property
    def enabled(self) -> bool:
        """True when any knob is set — i.e. this is a deliberately degraded system."""
        return bool(self.active())

    def hides_tool(self, tool: str) -> bool:
        """Return True when ``tool`` has been removed by ``FAULT_DROP_TOOL``."""
        return self.drop_tool is not None and self.drop_tool == tool
