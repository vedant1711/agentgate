"""Provider-independent token estimation.

Used for mock-mode accounting and budget projection when a provider does not return usage.
Deliberately a cheap heuristic (~4 characters per token) rather than a tokenizer dependency:
it must be identical on every machine, and a real tokenizer would vary by model family — which
would make replayed token counts non-reproducible.
"""

from __future__ import annotations

from collections.abc import Iterable

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text``.

    Args:
        text: Any string.

    Returns:
        At least 1 for non-empty input, 0 for the empty string.
    """
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def estimate_tokens_of(parts: Iterable[str]) -> int:
    """Sum :func:`estimate_tokens` over ``parts``."""
    return sum(estimate_tokens(part) for part in parts)
