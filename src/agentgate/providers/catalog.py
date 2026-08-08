"""What AgentGate knows about providers: routing, free-tier limits, and prices.

Prices are USD per **million** tokens. Free tiers and local Ollama are genuinely 0.00, but the
table still carries frontier prices so ``efficiency.est_cost_usd`` can project what the same
suite would cost at enterprise scale (B5) — the "minimal resources now, enterprise later"
argument needs a number, not a hand wave.

Rate limits are deliberately conservative: exceeding a free tier costs a run, under-using one
costs a few seconds. Override per provider via :class:`ProviderLimits`.
"""

from __future__ import annotations

import os
from typing import Final

from agentgate.schemas.common import FrozenModel
from agentgate.schemas.trajectory import TokenUsage

MOCK_PROVIDER: Final = "mock"


class ModelPrice(FrozenModel):
    """USD per million tokens."""

    prompt_per_mtok: float = 0.0
    completion_per_mtok: float = 0.0
    note: str = ""

    def cost(self, usage: TokenUsage) -> float:
        """Return the USD cost of ``usage`` at this price point."""
        return (
            usage.prompt_tokens * self.prompt_per_mtok
            + usage.completion_tokens * self.completion_per_mtok
        ) / 1_000_000.0


class ProviderLimits(FrozenModel):
    """Token-bucket configuration for one provider's free tier."""

    requests_per_minute: float = 60.0
    burst: int = 4
    tokens_per_minute: float = 0.0
    max_concurrency: int = 4


FREE: Final = ModelPrice(note="free tier / local — costs nothing, still counted")

PRICE_TABLE: Final[dict[str, ModelPrice]] = {
    # --- free tiers and local models (what AgentGate actually runs on) ---
    "groq/": FREE,
    "gemini/": FREE,
    "cerebras/": FREE,
    "openrouter/": ModelPrice(note="free-tier OpenRouter models only; paid slugs are rejected"),
    "ollama/": ModelPrice(note="local inference — electricity only"),
    "ollama_chat/": ModelPrice(note="local inference — electricity only"),
    "nvidia_nim/": ModelPrice(note="NVIDIA build.nvidia.com free tier"),
    "mock/": ModelPrice(note="deterministic fixtures"),
    # --- reference prices for enterprise projection only (never billed here) ---
    "gpt-4o": ModelPrice(prompt_per_mtok=2.50, completion_per_mtok=10.00, note="projection only"),
    "gpt-4o-mini": ModelPrice(
        prompt_per_mtok=0.15, completion_per_mtok=0.60, note="projection only"
    ),
    "claude-sonnet": ModelPrice(
        prompt_per_mtok=3.00, completion_per_mtok=15.00, note="projection only"
    ),
    "claude-haiku": ModelPrice(
        prompt_per_mtok=0.80, completion_per_mtok=4.00, note="projection only"
    ),
}

PROVIDER_LIMITS: Final[dict[str, ProviderLimits]] = {
    # Free-tier ceilings, set deliberately below the published limit. Exceeding a free tier
    # costs a whole run; under-using one costs a few seconds.
    "groq": ProviderLimits(requests_per_minute=28.0, burst=4, max_concurrency=4),
    "gemini": ProviderLimits(requests_per_minute=14.0, burst=2, max_concurrency=2),
    "nvidia_nim": ProviderLimits(requests_per_minute=36.0, burst=4, max_concurrency=4),
    "cerebras": ProviderLimits(requests_per_minute=28.0, burst=4, max_concurrency=4),
    "openrouter": ProviderLimits(requests_per_minute=18.0, burst=3, max_concurrency=3),
    # Local inference has no quota, but it is compute-bound: a second concurrent request on a
    # laptop makes both slower rather than finishing sooner.
    "ollama": ProviderLimits(requests_per_minute=600.0, burst=8, max_concurrency=1),
    "ollama_chat": ProviderLimits(requests_per_minute=600.0, burst=8, max_concurrency=1),
    MOCK_PROVIDER: ProviderLimits(requests_per_minute=1_000_000.0, burst=1024, max_concurrency=64),
}

DEFAULT_LIMITS: Final = ProviderLimits()

_KNOWN_PREFIXES: Final = (
    "groq",
    "gemini",
    "nvidia_nim",
    "cerebras",
    "openrouter",
    # LiteLLM routes Ollama tool-calling through `ollama_chat/`, not `ollama/`. Both are known
    # so a model id written either way lands on the right rate limiter.
    "ollama",
    "ollama_chat",
    MOCK_PROVIDER,
)


def provider_of(model: str) -> str:
    """Infer the provider from a model id.

    Args:
        model: A model id such as ``"groq/llama-3.3-70b"`` or ``"ollama/qwen3:4b"``.

    Returns:
        The provider slug, or ``"unknown"`` when the id carries no recognised prefix.
    """
    prefix, _, _ = model.partition("/")
    if prefix in _KNOWN_PREFIXES:
        return prefix
    return "unknown"


def limits_for(model: str) -> ProviderLimits:
    """Return the token-bucket configuration for ``model``'s provider."""
    return PROVIDER_LIMITS.get(provider_of(model), DEFAULT_LIMITS)


def price_for(model: str) -> ModelPrice:
    """Return the price entry for ``model``.

    Longest matching key wins, so ``"gpt-4o-mini"`` does not fall through to ``"gpt-4o"``.
    """
    if model in PRICE_TABLE:
        return PRICE_TABLE[model]
    matches = [key for key in PRICE_TABLE if key.endswith("/") and model.startswith(key)]
    matches += [key for key in PRICE_TABLE if not key.endswith("/") and key in model]
    if matches:
        return PRICE_TABLE[max(matches, key=len)]
    return ModelPrice(note="unpriced model — cost reported as 0.00")


def estimate_cost(model: str, usage: TokenUsage) -> float:
    """Return the projected USD cost of ``usage`` for ``model`` (B5)."""
    return price_for(model).cost(usage)


def ollama_base_url() -> str:
    """Return the Ollama endpoint, honouring ``OLLAMA_HOST``."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not host.startswith("http"):
        host = f"http://{host}"
    return host


def ollama_available(*, timeout_s: float = 0.75) -> bool:
    """Detect a running local Ollama daemon (A3.1's always-available fallback).

    Args:
        timeout_s: Connection timeout. Kept short so startup never stalls.

    Returns:
        True when ``/api/tags`` answers.
    """
    import urllib.error
    import urllib.request

    url = f"{ollama_base_url()}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return bool(200 <= response.status < 300)
    except (urllib.error.URLError, OSError, ValueError):
        return False
