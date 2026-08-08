"""Rubric and pairwise judges (D1, D2).

The rubric judge draws **J=3 samples at temperature 0.3** per item and keeps every draw, because
a judge is an instrument with error and a single draw hides that error entirely (C5). Malformed
JSON is retried and then *flagged* — never silently scored zero, which would turn a parsing bug
into a fabricated regression (H1).

The pairwise judge evaluates both slot orders and averages, which is the standard mitigation for
the 10-15 point position swing measured in MT-Bench and Wang et al. (arXiv:2305.17926). The
disagreement between orders is not discarded: it becomes the suite's **position-flip rate**, a
published number that says how much to trust this judge on this task type.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from agentgate.errors import ProviderError
from agentgate.judge.independence import check_independence
from agentgate.judge.rubrics import (
    OUTPUT_SCHEMA,
    PAIRWISE_SCHEMA,
    PAIRWISE_SYSTEM,
    RUBRICS,
    Rubric,
    normalise,
    render_pairwise,
)
from agentgate.judge.transcript import (
    JudgedItem,
    JudgeEntry,
    JudgeSample,
    JudgeTranscript,
    PairwiseEntry,
)
from agentgate.providers.client import LLMClient
from agentgate.providers.types import ChatMessage, ChatRequest
from agentgate.seeds import derive_seed

DEFAULT_JUDGE_MODEL = "mock/judge"
DEFAULT_CRITERIA = ("correctness", "completeness", "instruction_following", "coherence")


@dataclass(slots=True)
class JudgeConfig:
    """How the judge is run.

    Args:
        model: Judge model id. Pinned into ``agentgate.lock``.
        temperature: Non-zero by design — J draws at temperature 0 would all be identical and
            would report zero measurement error, which is a lie rather than a measurement.
        samples: J, the number of draws per item (D1).
        max_retries: Reparse attempts before an item is flagged.
        base_seed: Seeds per-draw request seeds so a judging pass is reproducible.
    """

    model: str = DEFAULT_JUDGE_MODEL
    temperature: float = 0.3
    samples: int = 3
    max_retries: int = 2
    base_seed: int = 20260101
    criteria: tuple[str, ...] = DEFAULT_CRITERIA
    max_concurrency: int = 4


@dataclass
class RubricJudge:
    """Runs the rubric criteria over a batch of items and returns a transcript.

    Args:
        client: Provider client. Its mode decides live/cache/replay/mock, exactly as for agents.
        config: Judge configuration.
        agent_model: Model the agent used, checked against the independence rule (D3).
        allow_self_judging: Explicit override of that rule.
    """

    client: LLMClient
    config: JudgeConfig = field(default_factory=JudgeConfig)
    agent_model: str = ""
    allow_self_judging: bool = False
    name: str = "rubric"

    def __post_init__(self) -> None:
        """Enforce judge/agent independence before any scoring happens."""
        self.independence_warning: str | None = None
        if self.agent_model:
            self.independence_warning = check_independence(
                self.agent_model, self.config.model, allow_self_judging=self.allow_self_judging
            )

    # -- scoring -----------------------------------------------------------

    async def score_item(self, criterion: str, item: JudgedItem) -> JudgeEntry:
        """Draw J samples for one (criterion, item).

        Args:
            criterion: A key of :data:`~agentgate.judge.rubrics.RUBRICS`.
            item: What to judge.

        Returns:
            The entry, flagged when no draw parsed.
        """
        rubric = RUBRICS[criterion]
        key = item.key_for(criterion)
        samples: list[JudgeSample] = []
        failures = 0
        for draw in range(self.config.samples):
            sample, draw_failures = await self._draw(rubric, item, draw)
            failures += draw_failures
            if sample is not None:
                samples.append(sample)
        return JudgeEntry(
            key=key,
            criterion=criterion,
            samples=samples,
            parse_failures=failures,
            flagged=not samples,
        )

    async def _draw(
        self, rubric: Rubric, item: JudgedItem, draw: int
    ) -> tuple[JudgeSample | None, int]:
        """One judge draw, with reparse retries."""
        failures = 0
        for attempt in range(self.config.max_retries + 1):
            request = ChatRequest(
                model=self.config.model,
                messages=[
                    ChatMessage(role="system", content=rubric.system_prompt()),
                    ChatMessage(
                        role="user",
                        content=rubric.render(
                            prompt=item.prompt,
                            response=item.response,
                            reference=item.reference,
                            contexts=list(item.contexts),
                        ),
                    ),
                ],
                temperature=self.config.temperature,
                seed=derive_seed(
                    self.config.base_seed,
                    rubric.criterion,
                    item.key_for(rubric.criterion),
                    draw,
                    attempt,
                ),
                response_format={"type": "json_object", "schema": OUTPUT_SCHEMA},
            )
            try:
                response = await self.client.complete(request)
            except ProviderError:
                failures += 1
                continue
            parsed = parse_rubric_reply(response.text)
            if parsed is None:
                failures += 1
                continue
            raw, reasoning, evidence = parsed
            return (
                JudgeSample(
                    raw_score=raw,
                    score=normalise(raw),
                    reasoning=reasoning,
                    evidence=evidence,
                    tokens=response.usage.total_tokens,
                ),
                failures,
            )
        return None, failures

    async def run(
        self, items: Sequence[JudgedItem], *, criteria: Sequence[str] | None = None
    ) -> JudgeTranscript:
        """Judge every item on every criterion.

        Args:
            items: Items to judge.
            criteria: Criteria to run; defaults to the config's.

        Returns:
            A transcript carrying every draw, with the independence warning attached when the
            rule was overridden.
        """
        wanted = list(criteria or self.config.criteria)
        transcript = JudgeTranscript(
            judge_name=self.name,
            judge_model=self.config.model,
            temperature=self.config.temperature,
            n_samples=self.config.samples,
            created_at=datetime.now(UTC),
        )
        if self.independence_warning:
            transcript.warnings.append(self.independence_warning)

        semaphore = asyncio.Semaphore(max(1, self.config.max_concurrency))

        async def one(criterion: str, item: JudgedItem) -> JudgeEntry:
            async with semaphore:
                return await self.score_item(criterion, item)

        jobs = [
            one(criterion, item)
            for criterion in wanted
            for item in items
            if _applicable(criterion, item)
        ]
        for entry in await asyncio.gather(*jobs):
            transcript.record(entry)

        flagged = transcript.flagged_items
        if flagged:
            transcript.warnings.append(
                f"{len(flagged)} item(s) produced unparseable judge output and were flagged; "
                f"they are excluded from judge-backed metrics rather than scored 0"
            )
        return transcript

    # -- pairwise ----------------------------------------------------------

    async def compare(
        self, prompt: str, candidate: str, baseline: str, *, key: str = ""
    ) -> PairwiseEntry:
        """Compare two responses in both slot orders and average (D2).

        Args:
            prompt: The shared task.
            candidate: The response under test — slot A forward, slot B swapped.
            baseline: The reference response.
            key: Identity for the transcript.

        Returns:
            The entry, from which flip and averaged score are derived.
        """
        forward = await self._compare_once(prompt, candidate, baseline, order=0)
        swapped = await self._compare_once(prompt, baseline, candidate, order=1)
        return PairwiseEntry(
            key=key or f"{hash((prompt, candidate, baseline)) & 0xFFFFFFFF:08x}",
            forward_winner=forward,
            swapped_winner=swapped,
        )

    async def _compare_once(self, prompt: str, slot_a: str, slot_b: str, *, order: int) -> str:
        request = ChatRequest(
            model=self.config.model,
            messages=[
                ChatMessage(role="system", content=PAIRWISE_SYSTEM),
                ChatMessage(role="user", content=render_pairwise(prompt, slot_a, slot_b)),
            ],
            temperature=self.config.temperature,
            seed=derive_seed(self.config.base_seed, "pairwise", prompt, order),
            response_format={"type": "json_object", "schema": PAIRWISE_SCHEMA},
        )
        response = await self.client.complete(request)
        return parse_pairwise_reply(response.text)


def _applicable(criterion: str, item: JudgedItem) -> bool:
    """Skip criteria whose inputs an item cannot supply."""
    rubric = RUBRICS[criterion]
    if rubric.needs_reference and not item.reference:
        return False
    return not (rubric.needs_contexts and not item.contexts)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> Any:
    """Pull a JSON object out of a judge reply, tolerating fences and surrounding prose."""
    candidate = text.strip()
    if candidate.startswith("```"):
        body = candidate.split("\n", 1)[1] if "\n" in candidate else ""
        candidate = body.rsplit("```", 1)[0].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None


def parse_rubric_reply(text: str) -> tuple[float, str, list[str]] | None:
    """Parse a rubric-judge reply into ``(raw score, reasoning, evidence)``.

    Args:
        text: Raw model output.

    Returns:
        ``None`` when the reply cannot be parsed or carries an out-of-range score — the caller
        retries and then flags, so a malformed reply never becomes a fabricated score.
    """
    payload = _extract_json(text)
    if not isinstance(payload, dict) or "score" not in payload:
        return None
    try:
        raw = float(payload["score"])
    except (TypeError, ValueError):
        return None
    if not 1.0 <= raw <= 5.0:
        return None
    evidence = payload.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    return raw, str(payload.get("reasoning", "")), [str(item) for item in evidence]


def parse_pairwise_reply(text: str) -> str:
    """Parse a pairwise reply into ``"A"``, ``"B"``, or ``"tie"``.

    An unparseable comparison is a ``tie``: refusing to guess a winner is the conservative
    choice, since a fabricated winner would move the win rate.
    """
    payload = _extract_json(text)
    if isinstance(payload, dict):
        winner = str(payload.get("winner", "")).strip().upper()
        if winner in ("A", "B"):
            return winner
        if winner == "TIE":
            return "tie"
    stripped = text.strip().upper()
    if stripped in ("A", "B"):
        return stripped
    return "tie"
