"""Fault-injection layer: the regressions AgentGate ships so it can be seen catching them."""

from agentgate.faults.config import ENV_PREFIX, FaultConfig
from agentgate.faults.signatures import (
    SIGNATURES,
    FaultSignature,
    MetricExpectation,
    scenario_names,
    signature,
)

__all__ = [
    "ENV_PREFIX",
    "SIGNATURES",
    "FaultConfig",
    "FaultSignature",
    "MetricExpectation",
    "scenario_names",
    "signature",
]
