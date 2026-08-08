"""The metric plugin contract (L4).

Every metric — deterministic, semantic, trajectory, RAG, safety, efficiency — implements one
protocol, so the statistics engine never needs to know which is which; it reads ``dtype`` and
applies the machinery Part C says is legal for that measurement type.

The most consequential rule here is that a metric whose requirements a task does not satisfy is
**skipped**, not scored zero. Scoring a missing reference as 0 would silently drag the suite mean
down and make a task-authoring gap look like an agent regression.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from agentgate.schemas.common import Direction, DType, MetricFamily, Requirement
from agentgate.schemas.results import MetricResult
from agentgate.schemas.task import ReferenceSpec, TaskSpec
from agentgate.schemas.trajectory import RunStatus, Trajectory


@runtime_checkable
class Embedder(Protocol):
    """Turns text into a vector. Local and deterministic by construction (A3.1)."""

    @property
    def name(self) -> str:
        """Identifier recorded next to every similarity number."""
        ...

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings."""
        ...


@runtime_checkable
class Judge(Protocol):
    """An LLM judge, as the metrics layer sees it.

    Declared here so judge-backed metrics can exist and be tested against a deterministic stub
    before the real bias-controlled judges land (Part D).
    """

    @property
    def name(self) -> str:
        """Identifier recorded next to every judge-backed number."""
        ...

    def score_criterion(
        self,
        criterion: str,
        prompt: str,
        response: str,
        *,
        reference: str = "",
        contexts: list[str] | None = None,
    ) -> JudgeVerdict:
        """Score one rubric criterion in [0, 1] with its per-sample variance."""
        ...

    def check_claims(self, claims: list[str], evidence: list[str]) -> list[bool]:
        """Return, per claim, whether the evidence supports it."""
        ...

    def extract_claims(self, text: str) -> list[str]:
        """Decompose ``text`` into atomic factual claims."""
        ...


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """One judge measurement: a mean over J samples, plus the spread across them (C5)."""

    value: float
    samples: tuple[float, ...] = ()
    variance: float | None = None
    cost_tokens: int = 0
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class MetricContext:
    """Services a metric may use, supplied by the engine rather than constructed by the metric."""

    judge: Judge | None = None
    embedder: Embedder | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def option(self, key: str, default: Any) -> Any:
        """Read a metric option with a fallback."""
        return self.options.get(key, default)


@dataclass(frozen=True, slots=True)
class ScoredSample:
    """One (task, repetition) pair, plus the services needed to score it."""

    task: TaskSpec
    trajectory: Trajectory
    run_id: str = ""
    context: MetricContext = field(default_factory=MetricContext)

    @property
    def reference(self) -> ReferenceSpec:
        """The task's declared ground truth."""
        return self.task.reference

    @property
    def answer(self) -> str:
        """The agent's final answer."""
        return self.trajectory.final_answer

    @property
    def system(self) -> str:
        """System-under-test label."""
        return self.trajectory.system

    @property
    def executed(self) -> bool:
        """False for units a budget halt prevented from running — those score nothing."""
        return self.trajectory.status is not RunStatus.BUDGET_EXHAUSTED

    def satisfied(self) -> set[Requirement]:
        """Which metric requirements this sample can actually meet."""
        reference = self.reference
        met: set[Requirement] = set()
        if reference.answer or reference.accepted_answers or reference.numeric_answer is not None:
            met.add(Requirement.REFERENCE_ANSWER)
        if reference.trajectory is not None:
            met.add(Requirement.REFERENCE_TRAJECTORY)
        if reference.goal_state:
            met.add(Requirement.GOAL_STATE)
        if self.trajectory.retrieved_contexts:
            met.add(Requirement.CONTEXTS)
        if reference.output_schema:
            met.add(Requirement.OUTPUT_SCHEMA)
        if self.trajectory.final_state is not None or self.trajectory.sandbox_events:
            met.add(Requirement.SANDBOX_EVENTS)
        if self.context.judge is not None:
            met.add(Requirement.JUDGE)
        if self.context.embedder is not None:
            met.add(Requirement.EMBEDDINGS)
        return met


@dataclass(frozen=True, slots=True)
class Scored:
    """What a metric's ``compute`` returns before the base class wraps it."""

    value: float | None
    detail: dict[str, Any] = field(default_factory=dict)
    cost_tokens: int = 0
    judge_samples: tuple[float, ...] = ()
    judge_variance: float | None = None

    @classmethod
    def skip(cls, reason: str) -> Scored:
        """Signal that this sample is not scoreable, with the reason recorded for the audit."""
        return cls(value=None, detail={"skipped": reason})


@runtime_checkable
class Metric(Protocol):
    """The uniform metric contract (Part B).

    The four descriptors are read-only properties so an implementation may expose them as
    ``ClassVar`` (the usual case), instance attributes, or computed properties — the engine and
    the registry only ever read them.
    """

    @property
    def name(self) -> str:
        """Unique metric name, e.g. ``trajectory.in_order_match``."""
        ...

    @property
    def family(self) -> MetricFamily:
        """Which Part B family this metric belongs to."""
        ...

    @property
    def direction(self) -> Direction:
        """Which way is better. Statistics direction-normalise before gating."""
        ...

    @property
    def dtype(self) -> DType:
        """Measurement type, which decides the legal statistical machinery (Part C)."""
        ...

    @property
    def requires(self) -> set[Requirement]:
        """Inputs a task must provide before this metric may score it."""
        ...

    def score(self, sample: ScoredSample) -> MetricResult:
        """Score one (task, repetition)."""
        ...


class BaseMetric(ABC):
    """Requirement checking, error containment, and result packaging for a metric.

    Subclasses implement :meth:`compute` and declare four class attributes. Everything a metric
    can do wrong — raising, returning out of range, being asked to score a sample it cannot —
    is handled once, here.
    """

    name: str = ""
    """Instance-level so a parameterised metric (e.g. ``single_tool_use[T]``) can name itself."""

    family: ClassVar[MetricFamily]
    direction: ClassVar[Direction] = "higher_is_better"
    dtype: ClassVar[DType] = "proportion"
    requires: ClassVar[set[Requirement]] = set()
    description: ClassVar[str] = ""

    @abstractmethod
    def compute(self, sample: ScoredSample) -> Scored:
        """Score ``sample``. Raising is allowed; the base class records the failure."""

    def applicable(self, sample: ScoredSample) -> str | None:
        """Return a skip reason, or ``None`` when this metric can score ``sample``."""
        if not sample.executed:
            return "unit was not executed (budget halt)"
        missing = self.requires - sample.satisfied()
        if missing:
            names = ", ".join(sorted(requirement.value for requirement in missing))
            return f"task does not provide: {names}"
        return None

    def score(self, sample: ScoredSample) -> MetricResult:
        """Score ``sample``, returning a skipped or errored result rather than raising."""
        skip_reason = self.applicable(sample)
        if skip_reason is not None:
            return self._result(sample, Scored.skip(skip_reason), status="skipped")
        try:
            outcome = self.compute(sample)
        except Exception as exc:
            return self._result(
                sample,
                Scored(value=None, detail={"exception": type(exc).__name__}),
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )
        if outcome.value is None:
            return self._result(sample, outcome, status="skipped")
        return self._result(sample, self._validated(outcome))

    def _validated(self, outcome: Scored) -> Scored:
        """Clamp proportions and binaries into range, recording any correction."""
        value = outcome.value
        if value is None or self.dtype in ("continuous", "count"):
            return outcome
        clamped = min(1.0, max(0.0, value))
        if clamped == value:
            return outcome
        detail = {**outcome.detail, "clamped_from": value}
        return Scored(
            value=clamped,
            detail=detail,
            cost_tokens=outcome.cost_tokens,
            judge_samples=outcome.judge_samples,
            judge_variance=outcome.judge_variance,
        )

    def _result(
        self,
        sample: ScoredSample,
        outcome: Scored,
        *,
        status: str = "ok",
        error: str | None = None,
    ) -> MetricResult:
        return MetricResult(
            metric=self.name,
            task_id=sample.task.id,
            rep=sample.trajectory.rep,
            system=sample.system,
            value=outcome.value,
            family=self.family,
            dtype=self.dtype,
            direction=self.direction,
            status=status,  # type: ignore[arg-type]
            error=error,
            detail=outcome.detail,
            judge_samples=list(outcome.judge_samples),
            judge_variance=outcome.judge_variance,
            cost_tokens=outcome.cost_tokens,
        )


def binary(value: bool) -> float:
    """Convert a boolean check into a binary metric value."""
    return 1.0 if value else 0.0


def ratio(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    """Safe division for proportion metrics.

    Args:
        numerator: Top of the fraction.
        denominator: Bottom of the fraction.
        default: Value when the denominator is zero — chosen per metric, because "no
            opportunities to be wrong" is not the same as "wrong".

    Returns:
        The ratio, or ``default``.
    """
    if denominator == 0:
        return default
    return numerator / denominator


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall; 0 when both are 0."""
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
