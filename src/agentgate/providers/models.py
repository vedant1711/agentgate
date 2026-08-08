"""The model catalogue: which models this harness can actually drive, and what they cost.

AgentGate is a *harness*, so the set of models it can evaluate is part of its value. This module
is the registry: every model is declared with its provider, whether it can call tools, whether it
runs locally, and roughly how big it is.

Two facts shape everything here, both measured rather than assumed:

* **Tool calling is not universal.** ``gemma3:4b`` refuses tool schemas outright. A model that
  cannot call tools cannot drive an agent, and the harness says so up front instead of failing
  three minutes into a run.
* **Speed varies by an order of magnitude.** ``llama3.2:3b`` answers a tool-calling turn in
  ~2.4s warm on an M4; the reasoning model ``qwen3:4b`` takes ~75s for the same turn. A 70-task
  suite at K=4 is roughly 1,120 calls, so the difference is 45 minutes versus a full day. That
  is why recording is a *background* activity and the gate always reads from cache.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final

from agentgate.providers.catalog import ollama_available, ollama_base_url

# Environment variable each provider's credentials live in. Ollama has none: it is local.
PROVIDER_KEYS: Final[dict[str, str]] = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "nvidia_nim": "NVIDIA_NIM_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

SIGNUP_URLS: Final[dict[str, str]] = {
    "groq": "https://console.groq.com/keys",
    "gemini": "https://aistudio.google.com/apikey",
    "nvidia_nim": "https://build.nvidia.com",
    "cerebras": "https://cloud.cerebras.ai",
    "openrouter": "https://openrouter.ai/keys",
    "ollama_chat": "https://ollama.com/download",
}


@dataclass(frozen=True, slots=True)
class ModelCard:
    """One model the harness knows how to drive.

    Args:
        model_id: The id passed to LiteLLM, e.g. ``"ollama_chat/llama3.2:3b"``.
        label: Short human name for reports and the leaderboard.
        provider: Provider slug, matching :func:`~agentgate.providers.catalog.provider_of`.
        params_b: Parameter count in billions; ``None`` when the provider does not say.
        supports_tools: Whether the model accepts tool schemas. A ``False`` here means the model
            can be judged but cannot *be* an agent.
        local: Runs on this machine, so no key, no quota, and no network.
        approx_s_per_call: Measured seconds per tool-calling turn where we have a number.
        notes: Anything a user should know before picking it.
    """

    model_id: str
    label: str
    provider: str
    params_b: float | None = None
    supports_tools: bool = True
    local: bool = False
    approx_s_per_call: float | None = None
    notes: str = ""

    @property
    def key_env(self) -> str | None:
        """Environment variable holding this provider's key, or ``None`` when local."""
        return PROVIDER_KEYS.get(self.provider)

    @property
    def key_present(self) -> bool:
        """True when this model can be reached: local, or its key is set."""
        if self.local:
            return True
        env = self.key_env
        return bool(env and os.environ.get(env))


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------

CATALOG: Final[tuple[ModelCard, ...]] = (
    # --- Local, via Ollama. No key, no quota, no network. ------------------
    ModelCard(
        model_id="ollama_chat/llama3.2:3b",
        label="Llama 3.2 3B",
        provider="ollama_chat",
        params_b=3.2,
        local=True,
        approx_s_per_call=2.4,
        notes=(
            "Measured on an M4: ~2.4s/turn warm (~7s cold). On tau2_retail (111 tasks, K=3, "
            "104 min) it solves 18% of tasks [0.108, 0.252] while scoring 0.99 on judged "
            "coherence — fluent, efficient, and wrong. That gap is the single most useful "
            "number the harness has produced."
        ),
    ),
    ModelCard(
        model_id="ollama_chat/qwen3:4b",
        label="Qwen3 4B",
        provider="ollama_chat",
        params_b=4.0,
        local=True,
        approx_s_per_call=75.0,
        notes="Correct tool calls but ~10x slower: it is a reasoning model and thinks at length.",
    ),
    ModelCard(
        model_id="ollama_chat/qwen2.5:7b",
        label="Qwen2.5 7B",
        provider="ollama_chat",
        params_b=7.6,
        local=True,
        notes="Stronger than the 3B models; needs ~5GB of RAM headroom.",
    ),
    ModelCard(
        model_id="ollama_chat/gemma3:4b",
        label="Gemma 3 4B",
        provider="ollama_chat",
        params_b=4.3,
        supports_tools=False,
        local=True,
        notes="Cannot call tools — Ollama rejects the schema. Usable as a judge, not as an agent.",
    ),
    # --- Groq: fastest free tier, tight daily caps. ------------------------
    ModelCard(
        model_id="groq/llama-3.3-70b-versatile",
        label="Llama 3.3 70B",
        provider="groq",
        params_b=70.0,
        notes="Far stronger than anything local. Free tier is fast but daily-capped.",
    ),
    ModelCard(
        model_id="groq/llama-3.1-8b-instant",
        label="Llama 3.1 8B",
        provider="groq",
        params_b=8.0,
        notes="Cheap and quick; a good candidate for the 'cost-driven downgrade' comparison.",
    ),
    # --- Google AI Studio. -------------------------------------------------
    ModelCard(
        model_id="gemini/gemini-2.0-flash",
        label="Gemini 2.0 Flash",
        provider="gemini",
        notes="Generous free tier. A different model family, which matters for judge independence.",
    ),
    # --- NVIDIA NIM: 100+ models, 40 req/min, no card. ---------------------
    ModelCard(
        model_id="nvidia_nim/meta/llama-3.3-70b-instruct",
        label="Llama 3.3 70B (NIM)",
        provider="nvidia_nim",
        params_b=70.0,
        notes="Same weights as Groq's, different host — useful for provider-effect checks.",
    ),
    ModelCard(
        model_id="nvidia_nim/qwen/qwen2.5-coder-32b-instruct",
        label="Qwen2.5 Coder 32B (NIM)",
        provider="nvidia_nim",
        params_b=32.0,
        notes="Strong structured-output model; good judge candidate.",
    ),
    # --- The offline stand-in. --------------------------------------------
    ModelCard(
        model_id="mock/agent",
        label="Deterministic brain",
        provider="mock",
        supports_tools=True,
        local=True,
        approx_s_per_call=0.005,
        notes="Not a model. A rule-following stand-in so the pipeline runs offline and free.",
    ),
)

BY_ID: Final[dict[str, ModelCard]] = {card.model_id: card for card in CATALOG}


def get_card(model_id: str) -> ModelCard | None:
    """Look up a model card, or ``None`` when the id is not in the catalogue."""
    return BY_ID.get(model_id)


def agent_capable() -> list[ModelCard]:
    """Models that can actually drive an agent: they support tool calling."""
    return [card for card in CATALOG if card.supports_tools and card.provider != "mock"]


def judge_capable() -> list[ModelCard]:
    """Models usable as judges. Tool calling is not required to grade text."""
    return [card for card in CATALOG if card.provider != "mock"]


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    """Whether a provider can be reached right now, and what to do if not."""

    provider: str
    reachable: bool
    reason: str
    models: list[ModelCard] = field(default_factory=list)
    signup_url: str = ""

    @property
    def n_agent_models(self) -> int:
        """How many of this provider's models can drive an agent."""
        return sum(1 for card in self.models if card.supports_tools)


def provider_status(provider: str, *, check_network: bool = True) -> ProviderStatus:
    """Report whether one provider is usable.

    Args:
        provider: Provider slug.
        check_network: Probe the local Ollama daemon. Skipped in tests.

    Returns:
        The status, with an actionable reason when it is not reachable.
    """
    models = [card for card in CATALOG if card.provider == provider]
    signup = SIGNUP_URLS.get(provider, "")

    if provider == "mock":
        return ProviderStatus(provider, True, "always available (no network)", models, signup)

    if provider == "ollama_chat":
        if not check_network:
            return ProviderStatus(provider, True, "not probed", models, signup)
        if ollama_available():
            return ProviderStatus(
                provider, True, f"daemon responding at {ollama_base_url()}", models, signup
            )
        return ProviderStatus(
            provider,
            False,
            "no Ollama daemon — run `ollama serve`, then `ollama pull llama3.2:3b`",
            models,
            signup,
        )

    env = PROVIDER_KEYS.get(provider)
    if env and os.environ.get(env):
        return ProviderStatus(provider, True, f"{env} is set", models, signup)
    return ProviderStatus(
        provider,
        False,
        f"{env} is not set — get a free key at {signup}" if env else "unknown provider",
        models,
        signup,
    )


def all_provider_status(*, check_network: bool = True) -> list[ProviderStatus]:
    """Status for every provider in the catalogue, in name order."""
    providers = sorted({card.provider for card in CATALOG})
    return [provider_status(name, check_network=check_network) for name in providers]


def usable_agent_models(*, check_network: bool = True) -> list[ModelCard]:
    """Every tool-capable model that is reachable right now.

    This is what the continuous harness iterates over: it evaluates what it *can* evaluate,
    and records why it skipped the rest rather than silently narrowing its own scope.
    """
    reachable = {
        status.provider
        for status in all_provider_status(check_network=check_network)
        if status.reachable
    }
    return [card for card in agent_capable() if card.provider in reachable]


def describe_throughput(card: ModelCard, *, n_tasks: int, k: int, calls_per_task: int = 4) -> str:
    """Estimate how long a full recording run would take on this model.

    The harness is honest about this because it is the single biggest practical constraint:
    a full suite against a local model is measured in hours, which is why recording happens in
    the background and the gate always replays from cache.
    """
    total_calls = n_tasks * k * calls_per_task
    if card.approx_s_per_call is None:
        return f"{total_calls:,} model calls (rate unmeasured for this model)"
    seconds = total_calls * card.approx_s_per_call
    if seconds < 90:
        return f"{total_calls:,} calls, about {seconds:.0f}s"
    if seconds < 5400:
        return f"{total_calls:,} calls, about {seconds / 60:.0f} min"
    return f"{total_calls:,} calls, about {seconds / 3600:.1f} hours"
