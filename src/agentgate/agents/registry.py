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
from agentgate.agents.tau2_agent import Tau2RetailAgent
from agentgate.agents.tool_agent import ToolAgent
from agentgate.errors import ConfigError
from agentgate.faults.config import FaultConfig
from agentgate.providers.client import DEFAULT_CACHE_PATH, ClientConfig, LLMClient
from agentgate.providers.mock import Handler, MockTransport
from agentgate.providers.types import ChatRequest, ChatResponse
from agentgate.schemas.common import ProviderMode
from agentgate.schemas.results import BudgetSpec

AGENT_CLASSES: dict[str, type[BaseAgent]] = {
    ToolAgent.name: ToolAgent,
    RagAgent.name: RagAgent,
    PlanAgent.name: PlanAgent,
    Tau2RetailAgent.name: Tau2RetailAgent,
}
"""Reference agents by id."""

BRAINS: dict[str, Callable[[ChatRequest], ChatResponse]] = {
    ToolAgent.name: CrmBrain(),
    RagAgent.name: RagBrain(),
    PlanAgent.name: PlanBrain(),
    # tau2_retail_agent has no brain on purpose: a fabricated trajectory over real benchmark
    # tasks would look authoritative and mean nothing. It refuses to run in mock mode.
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
        known = ", ".join(sorted(BRAINS))
        msg = (
            f"no deterministic brain for {agent!r}; agents with one: {known}. "
            f"Agents built on published benchmarks require a real model."
        )
        raise ConfigError(msg)
    return BRAINS[agent]


MOCK_MODEL_PREFIX = "mock/"
"""Models under this prefix are served by the reference agent's deterministic brain."""


def build_client(
    agent: str,
    *,
    mode: ProviderMode = ProviderMode.MOCK,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
    namespace: str = "",
    budget: BudgetSpec | None = None,
    model: str = "mock/agent",
) -> LLMClient:
    """Build a provider client for ``agent``.

    A ``mock/*`` model is served by the agent's deterministic brain; anything else goes to the
    real provider stack. In ``cache`` mode the brain's replies are written through to SQLite,
    which is how the demo's replay fixtures are produced without ever touching a provider.

    Args:
        agent: Registered agent id.
        mode: Execution mode.
        cache_path: SQLite cache location.
        namespace: Cache-key salt for prompt revisions.
        budget: Run-level caps enforced on every provider call.
        model: Model id, which decides whether the brain or a real provider answers.

    Returns:
        A configured client.
    """
    config = ClientConfig(
        mode=mode,
        cache_path=":memory:" if mode is ProviderMode.MOCK else cache_path,
        namespace=namespace,
        budget=budget or BudgetSpec(),
    )
    transport = (
        MockTransport(brain_for(agent), provider="mock")
        if model.startswith(MOCK_MODEL_PREFIX) and agent in BRAINS
        else None
    )
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
    budget: BudgetSpec | None = None,
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
        budget: Run-level caps enforced on every provider call.
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
    agent_config = config or AgentConfig()
    agent_config.system = system
    resolved_client = client or build_client(
        agent, mode=mode, cache_path=cache_path, budget=budget, model=agent_config.model
    )

    if issubclass(agent_cls, Tau2RetailAgent):
        return agent_cls(
            client=resolved_client, config=agent_config, faults=faults or FaultConfig()
        )
    if issubclass(agent_cls, RagAgent | PlanAgent):
        return agent_cls(
            client=resolved_client,
            config=agent_config,
            faults=faults or FaultConfig(),
            corpus=corpus or WikiCorpus(),
        )
    return agent_cls(client=resolved_client, config=agent_config, faults=faults or FaultConfig())
