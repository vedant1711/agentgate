"""Provider layer: one client, four modes, zero paid resources.

Public surface for everything above L1. Import from here rather than from submodules so the
LiteLLM dependency stays lazy.
"""

from agentgate.providers.budget import BudgetTracker
from agentgate.providers.cache import CACHE_SCHEMA_VERSION, ResponseCache
from agentgate.providers.catalog import (
    PRICE_TABLE,
    PROVIDER_LIMITS,
    ModelPrice,
    ProviderLimits,
    estimate_cost,
    limits_for,
    ollama_available,
    ollama_base_url,
    price_for,
    provider_of,
)
from agentgate.providers.client import DEFAULT_CACHE_PATH, ClientConfig, LLMClient
from agentgate.providers.mock import MockTransport, ScriptedTransport, echo_handler
from agentgate.providers.ratelimit import RateLimiterRegistry, TokenBucket
from agentgate.providers.retry import RetryOutcome, RetryPolicy, backoff_delay, with_retry
from agentgate.providers.tokens import estimate_tokens, estimate_tokens_of
from agentgate.providers.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ClientStats,
    ToolCallRequest,
    ToolSpec,
    Transport,
)

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_CACHE_PATH",
    "PRICE_TABLE",
    "PROVIDER_LIMITS",
    "BudgetTracker",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ClientConfig",
    "ClientStats",
    "LLMClient",
    "MockTransport",
    "ModelPrice",
    "ProviderLimits",
    "RateLimiterRegistry",
    "ResponseCache",
    "RetryOutcome",
    "RetryPolicy",
    "ScriptedTransport",
    "TokenBucket",
    "ToolCallRequest",
    "ToolSpec",
    "Transport",
    "backoff_delay",
    "echo_handler",
    "estimate_cost",
    "estimate_tokens",
    "estimate_tokens_of",
    "limits_for",
    "ollama_available",
    "ollama_base_url",
    "price_for",
    "provider_of",
    "with_retry",
]
