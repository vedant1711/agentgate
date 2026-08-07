"""Provider-facing request/response types.

These are deliberately provider-agnostic (A3.2): LiteLLM, Ollama, and the mock transport all
speak this shape, so swapping the agent, judge, or embedding model is a config change rather
than a code change.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import Field

from agentgate.schemas.common import FrozenModel, canonical_dumps, sha256_hex
from agentgate.schemas.trajectory import TokenUsage

Role = Literal["system", "user", "assistant", "tool"]


class ToolCallRequest(FrozenModel):
    """A model's request to invoke a tool."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolSpec(FrozenModel):
    """A tool exposed to the model, described by JSON Schema."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}},
        description="JSON Schema for the tool's arguments.",
    )

    def to_openai(self) -> dict[str, Any]:
        """Render in the OpenAI/LiteLLM function-calling shape."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ChatMessage(FrozenModel):
    """One message in a conversation."""

    role: Role
    content: str = ""
    name: str | None = None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    tool_call_id: str | None = None

    def to_openai(self) -> dict[str, Any]:
        """Render in the OpenAI/LiteLLM message shape."""
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": canonical_dumps(call.arguments)},
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload


class ChatRequest(FrozenModel):
    """A single completion request.

    :meth:`cache_key` is the SQLite cache's primary key and therefore the definition of "the
    same request" for replay determinism (A3.6).
    """

    model: str
    messages: list[ChatMessage]
    tools: list[ToolSpec] = Field(default_factory=list)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    seed: int | None = None
    response_format: dict[str, Any] | None = None
    stop: list[str] = Field(default_factory=list)
    n: int = Field(default=1, ge=1)
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific knobs; participates in the cache key."
    )

    def cache_payload(self, namespace: str = "") -> dict[str, Any]:
        """Return the exact structure the cache key is computed over."""
        return {
            "namespace": namespace,
            "model": self.model,
            "messages": [message.model_dump(mode="json") for message in self.messages],
            "tools": [tool.model_dump(mode="json") for tool in self.tools],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "response_format": self.response_format,
            "stop": self.stop,
            "n": self.n,
            "extra": self.extra,
        }

    def cache_key(self, namespace: str = "") -> str:
        """Stable key over (model, messages, tools, params, seed) — A3.6."""
        return sha256_hex(canonical_dumps(self.cache_payload(namespace)))

    def prompt_hash(self) -> str:
        """Hash of the messages alone, recorded on the trajectory's ``llm_call`` step."""
        return sha256_hex(
            canonical_dumps([message.model_dump(mode="json") for message in self.messages])
        )[:16]


class ChatResponse(FrozenModel):
    """A completion result, normalised across providers.

    ``latency_ms`` is recorded at generation time and *restored* on cache hits so that latency
    metrics are reproducible in replay mode rather than measuring cache-lookup speed.
    """

    text: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)
    finish_reason: str = "stop"
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str = ""
    provider: str = ""
    latency_ms: float = Field(default=0.0, ge=0.0)
    cached: bool = False
    response_id: str | None = None

    def with_cache_flag(self, *, cached: bool) -> ChatResponse:
        """Return a copy with the cache flag set."""
        return self.model_copy(update={"cached": cached})


@runtime_checkable
class Transport(Protocol):
    """The thin seam every backend implements (A3.2)."""

    name: str

    async def complete(self, request: ChatRequest) -> ChatResponse:
        """Execute one completion request."""
        ...


class ClientStats(FrozenModel):
    """Per-run provider bookkeeping surfaced in the run summary."""

    requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retries: int = 0
    rate_limit_events: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    degraded_to_replay: bool = False

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens across the run."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from cache; 0.0 when nothing was looked up."""
        lookups = self.cache_hits + self.cache_misses
        return self.cache_hits / lookups if lookups else 0.0
