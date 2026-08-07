"""Agent registry and offline wiring.

The three reference agents are *specimens*, not products: their job is to be run thousands of
times for free so the statistics and gate layers have something honest to chew on. This module
wires each one to its deterministic brain, so ``mock`` runs need no keys and ``cache`` runs
produce genuine replay fixtures without a single network call.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from agentgate.agents.brains import CrmBrain, PlanBrain, RagBrain
from agentgate.agents.corpus import WikiCorpus
from agentgate.agents.plan_agent import PlanAgent
from agentgate.agents.protocol import AgentConfig, BaseAgent
from agentgate.agents.rag_agent import RagAgent
from agentgate.agents.tool_agent import ToolAgent
from agentgate.errors import ConfigError
from agentgate.faults.config import FaultConfig
from agentgate.providers.client import DEFAULT_CACHE_PATH, ClientConfig, LLMClient
from agentgate.providers.mock import Handler, MockTransport
from agentgate.providers.types import ChatRequest, ChatResponse
from agentgate.schemas.common import ProviderMode

AGENT_CLASSES: dict[str, type[BaseAgent]] = {
    ToolAgent.name: ToolAgent,
    RagAgent.name: RagAgent,
    PlanAgent.name: PlanAgent,
}
"""Reference agents by id."""

BRAINS: dict[str, Callable[[ChatRequest], ChatResponse]] = {
    ToolAgent.name: CrmBrain(),
    RagAgent.name: RagBrain(),
    PlanAgent.name: PlanBrain(),
}
"""The deterministic policy that stands in for a model in offline modes."""


def agent_names() -> list[str]:
    """Return every registered agent id, sorted."""
    return sorted(AGENT_CLASSES)


def brain_for(agent: str) -> Handler:
    """Return the deterministic brain for ``agent``.

    Args:
        agent: Registered agent id.

    Returns:
        A mock handler.

    Raises:
        ConfigError: When ``agent`` is not registered.
    """
    if agent not in BRAINS:
        msg = f"unknown agent {agent!r}; known agents: {', '.join(agent_names())}"
        raise ConfigError(msg)
    return BRAINS[agent]


def build_client(
    agent: str,
    *,
    mode: ProviderMode = ProviderMode.MOCK,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    namespace: str = "",
) -> LLMClient:
    """Build a provider client wired to ``agent``'s deterministic brain.

    In ``cache`` mode the brain's replies are written through to SQLite, which is how the demo's
    replay fixtures are produced without ever touching a provider.

    Args:
        agent: Registered agent id.
        mode: Execution mode.
        cache_path: SQLite cache location.
        namespace: Cache-key salt for prompt revisions.

    Returns:
        A configured client.
    """
    config = ClientConfig(
        mode=mode,
        cache_path=":memory:" if mode is ProviderMode.MOCK else cache_path,
        namespace=namespace,
    )
    transport = MockTransport(brain_for(agent), provider="mock")
    return LLMClient(config, transport=transport)


def build_agent(
    agent: str,
    *,
    client: LLMClient | None = None,
    mode: ProviderMode = ProviderMode.MOCK,
    faults: FaultConfig | None = None,
    config: AgentConfig | None = None,
    corpus: WikiCorpus | None = None,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    system: str = "baseline",
) -> BaseAgent:
    """Construct a reference agent.

    Args:
        agent: Registered agent id.
        client: Provider client. Built from ``mode`` when omitted.
        mode: Execution mode used to build a client.
        faults: Active regression knobs.
        config: Model/loop configuration.
        corpus: Wiki corpus for the retrieval agents.
        cache_path: SQLite cache location for non-mock modes.
        system: System-under-test label recorded on trajectories.

    Returns:
        An agent implementing :class:`~agentgate.agents.protocol.AgentUnderTest`.

    Raises:
        ConfigError: When ``agent`` is not registered.
    """
    if agent not in AGENT_CLASSES:
        msg = f"unknown agent {agent!r}; known agents: {', '.join(agent_names())}"
        raise ConfigError(msg)
    agent_cls = AGENT_CLASSES[agent]
    resolved_client = client or build_client(agent, mode=mode, cache_path=cache_path)
    agent_config = config or AgentConfig()
    agent_config.system = system

    if issubclass(agent_cls, RagAgent | PlanAgent):
        return agent_cls(
            client=resolved_client,
            config=agent_config,
            faults=faults or FaultConfig(),
            corpus=corpus or WikiCorpus(),
        )
    return agent_cls(client=resolved_client, config=agent_config, faults=faults or FaultConfig())
