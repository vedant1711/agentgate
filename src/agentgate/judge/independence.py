"""The judge/agent independence rule (D3).

LLM evaluators recognise and favour their own generations (Panickssery et al.,
arXiv:2410.21819). A judge from the same model family as the agent is therefore not an
independent instrument, and a gate backed by one is measuring kinship as much as quality.

AgentGate **default-denies** that configuration. The override exists — sometimes one free tier
is all you have — but it prints a warning into every report that used it, so the caveat travels
with the number instead of living in a config file nobody reads.
"""

from __future__ import annotations

import re
from typing import Final

from agentgate.errors import JudgeIndependenceError

FAMILY_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "llama": re.compile(r"llama", re.IGNORECASE),
    "qwen": re.compile(r"qwen", re.IGNORECASE),
    "gemma": re.compile(r"gemma", re.IGNORECASE),
    "gemini": re.compile(r"gemini", re.IGNORECASE),
    "gpt": re.compile(r"\bgpt|\bo[134]\b", re.IGNORECASE),
    "claude": re.compile(r"claude", re.IGNORECASE),
    "mistral": re.compile(r"mistral|mixtral", re.IGNORECASE),
    "deepseek": re.compile(r"deepseek", re.IGNORECASE),
    "phi": re.compile(r"\bphi-?\d", re.IGNORECASE),
    "command": re.compile(r"command-?r", re.IGNORECASE),
    "mock": re.compile(r"^mock/", re.IGNORECASE),
}
"""Model-family detectors, checked in order. Unknown ids report ``"unknown"``."""


def model_family(model_id: str) -> str:
    """Infer a model's family from its id.

    Args:
        model_id: Provider-qualified model id, e.g. ``"groq/llama-3.3-70b"``.

    Returns:
        A family slug, or ``"unknown"`` when nothing matches. Two ``"unknown"`` models are
        *not* treated as the same family — guessing kinship from ignorance would block valid
        configurations.
    """
    if model_id.lower().startswith("mock/"):
        return "mock"
    # Match against the model name only. The provider prefix is not part of the family, and
    # matching it would classify "ollama/qwen3" as llama because "ollama" contains "llama".
    _, _, name = model_id.partition("/")
    haystack = name or model_id
    for family, pattern in FAMILY_PATTERNS.items():
        if family == "mock":
            continue
        if pattern.search(haystack):
            return family
    return "unknown"


def same_family(agent_model: str, judge_model: str) -> bool:
    """True when both models are recognisably from the same family."""
    agent = model_family(agent_model)
    judge = model_family(judge_model)
    return agent == judge and agent != "unknown"


def check_independence(
    agent_model: str, judge_model: str, *, allow_self_judging: bool = False
) -> str | None:
    """Enforce the independence rule.

    Args:
        agent_model: Model the agent under test used.
        judge_model: Model backing the judge.
        allow_self_judging: Explicit override.

    Returns:
        A warning string to print in every report when the rule was overridden, else ``None``.

    Raises:
        JudgeIndependenceError: When the families match and no override was given.
    """
    if not same_family(agent_model, judge_model):
        return None
    family = model_family(agent_model)
    if not allow_self_judging:
        msg = (
            f"judge model {judge_model!r} and agent model {agent_model!r} are both in the "
            f"{family!r} family. Self-preference bias makes this judge non-independent "
            f"(D3). Choose a different judge family, or set allow_self_judging: true in "
            f"gate.yaml and accept the warning on every report."
        )
        raise JudgeIndependenceError(msg)
    return (
        f"SELF-JUDGING: judge ({judge_model}) and agent ({agent_model}) share the {family!r} "
        f"family. Scores are inflated by self-preference bias to an unknown degree."
    )
