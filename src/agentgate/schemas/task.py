"""Task and suite specifications (Part F3 of the spec).

Suites are versioned, content-hashed YAML. The gate refuses to compare runs whose suite
content hashes differ, because a paired test across different task instances is not a paired
test at all (C2).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import Field, computed_field, model_validator

from agentgate.schemas.common import SCHEMA_VERSION, FrozenModel


class Difficulty(StrEnum):
    """Author-declared task difficulty, used for stratified reporting only."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ArgComparator(FrozenModel):
    """How a single tool-call argument is compared against its reference value (B2)."""

    kind: Literal["exact", "numeric", "regex", "semantic", "ignore"] = "exact"
    tolerance: float = Field(
        default=0.0, ge=0.0, description="Absolute tolerance for kind='numeric'."
    )
    pattern: str | None = Field(default=None, description="Regex source for kind='regex'.")
    threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Cosine floor for kind='semantic'."
    )
    case_sensitive: bool = True

    @model_validator(mode="after")
    def _check_pattern(self) -> Self:
        if self.kind == "regex" and not self.pattern:
            msg = "ArgComparator(kind='regex') requires a `pattern`"
            raise ValueError(msg)
        return self


class ReferenceStep(FrozenModel):
    """One step of a gold trajectory.

    A step lists *allowed alternatives* rather than a single tool, so a task can declare
    ``search_web`` and ``search_docs`` equivalent. Steps carrying the same ``group`` may be
    satisfied in any relative order; ``optional`` steps never count against recall.
    """

    tools: list[str] = Field(min_length=1, description="Interchangeable tool names for this step.")
    args: dict[str, Any] | None = Field(
        default=None,
        description="Expected argument subset. Keys absent here are not scored.",
    )
    arg_comparators: dict[str, ArgComparator] = Field(default_factory=dict)
    optional: bool = False
    group: str | None = Field(
        default=None, description="Unordered-group id; steps sharing an id may occur in any order."
    )
    label: str | None = None

    def matches_tool(self, tool: str) -> bool:
        """Return True when ``tool`` satisfies this step's allowed-alternative set."""
        return tool in self.tools


class ReferenceTrajectory(FrozenModel):
    """Gold trajectory for a task: an ordered list of steps plus matching policy."""

    steps: list[ReferenceStep] = Field(default_factory=list)
    allow_extra_calls: bool = Field(
        default=True,
        description="If False, any unreferenced predicted call breaks in_order/any_order match.",
    )
    required_tools: list[str] = Field(
        default_factory=list,
        description="Tools scored by trajectory.single_tool_use[T].",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def required_step_count(self) -> int:
        """Number of non-optional reference steps (the denominator for recall)."""
        return sum(1 for step in self.steps if not step.optional)


class InjectionProbe(FrozenModel):
    """A planted indirect prompt injection and the observable that proves compliance (B4)."""

    payload: str = Field(description="Adversarial instruction planted in a tool output/document.")
    planted_in: str = Field(description="Tool name or document id carrying the payload.")
    compliance_signal: str = Field(
        description="Substring/tool name whose presence proves the agent obeyed the injection."
    )
    signal_kind: Literal["output_substring", "tool_call", "regex"] = "output_substring"


class ReferenceSpec(FrozenModel):
    """Everything a task declares as ground truth.

    Which metrics are eligible for a task is derived from which of these fields are populated
    (see :class:`agentgate.schemas.common.Requirement`).
    """

    answer: str | None = None
    accepted_answers: list[str] = Field(
        default_factory=list, description="Additional strings that count as exact matches."
    )
    numeric_answer: float | None = None
    numeric_tolerance: float = Field(default=0.0, ge=0.0)
    goal_state: dict[str, Any] | None = Field(
        default=None, description="tau-bench-style annotated end state for the sandbox checker."
    )
    trajectory: ReferenceTrajectory | None = None
    contexts: list[str] = Field(
        default_factory=list, description="Gold retrieval contexts for the RAG family."
    )
    output_schema: dict[str, Any] | None = Field(
        default=None, description="JSON Schema for outcome.schema_compliant."
    )
    forbidden_tools: list[str] = Field(default_factory=list)
    canary_pii: list[str] = Field(
        default_factory=list, description="Planted PII strings that must never reach the output."
    )
    injection: InjectionProbe | None = None
    unanswerable: bool = Field(
        default=False, description="Abstention check: the correct behaviour is to refuse."
    )


class CheckerSpec(FrozenModel):
    """Programmatic success checker for ``outcome.task_success`` (B1)."""

    name: str = Field(description="Registered checker id, e.g. 'goal_state' or 'answer_match'.")
    params: dict[str, Any] = Field(default_factory=dict)


class TaskSpec(FrozenModel):
    """A single evaluation task instance."""

    id: str = Field(pattern=r"^[a-zA-Z0-9._:-]+$")
    cluster_id: str = Field(
        description="Tasks sharing a cluster are statistically dependent (C1.3). "
        "Use the scenario/template/document they were generated from."
    )
    template: str | None = Field(
        default=None, description="Template name this task was paraphrased from."
    )
    inputs: dict[str, Any] = Field(
        description="Payload handed to the agent. Must contain at least one of prompt/question."
    )
    reference: ReferenceSpec = Field(default_factory=ReferenceSpec)
    checker: CheckerSpec | None = None
    tags: list[str] = Field(default_factory=list)
    difficulty: Difficulty = Difficulty.MEDIUM
    k: int | None = Field(default=None, ge=1, description="Per-task override of suite K.")
    max_steps: int = Field(default=12, ge=1)
    timeout_s: float = Field(default=120.0, gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_a_prompt(self) -> Self:
        if not any(key in self.inputs for key in ("prompt", "question", "instruction")):
            msg = f"task {self.id!r}: inputs must contain one of prompt/question/instruction"
            raise ValueError(msg)
        return self

    @property
    def prompt(self) -> str:
        """Return the task's natural-language instruction."""
        for key in ("prompt", "question", "instruction"):
            value = self.inputs.get(key)
            if isinstance(value, str):
                return value
        msg = f"task {self.id!r} has no textual prompt"  # pragma: no cover - guarded above
        raise ValueError(msg)


class SuiteSpec(FrozenModel):
    """A versioned collection of tasks.

    The ``content_hash`` covers every task and the suite's own identity, so any edit to a task
    invalidates cross-version comparison (F3).
    """

    schema_version: int = SCHEMA_VERSION
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    version: str = Field(default="0.1.0", description="Semantic version of the suite content.")
    description: str = ""
    default_k: int = Field(default=4, ge=1, description="Repetitions per task (E2).")
    agent: str = Field(default="tool_agent", description="Reference agent this suite targets.")
    tasks: list[TaskSpec] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_task_ids(self) -> Self:
        seen: set[str] = set()
        for task in self.tasks:
            if task.id in seen:
                msg = f"suite {self.name!r}: duplicate task id {task.id!r}"
                raise ValueError(msg)
            seen.add(task.id)
        return self

    @property
    def task_ids(self) -> list[str]:
        """Task ids in declaration order."""
        return [task.id for task in self.tasks]

    @property
    def cluster_ids(self) -> list[str]:
        """Cluster id per task, aligned with :attr:`task_ids`."""
        return [task.cluster_id for task in self.tasks]

    @property
    def n_clusters(self) -> int:
        """Number of distinct clusters in the suite."""
        return len({task.cluster_id for task in self.tasks})

    @property
    def has_clusters(self) -> bool:
        """True when tasks are grouped, i.e. clustered SEs are required (C1.3)."""
        return self.n_clusters < len(self.tasks)

    def task(self, task_id: str) -> TaskSpec:
        """Look up a task by id.

        Args:
            task_id: Identifier declared in the suite.

        Returns:
            The matching :class:`TaskSpec`.

        Raises:
            KeyError: If no task carries that id.
        """
        for task in self.tasks:
            if task.id == task_id:
                return task
        msg = f"task {task_id!r} not in suite {self.name!r}"
        raise KeyError(msg)


class SuiteRef(FrozenModel):
    """Immutable pointer to the exact suite content a run consumed."""

    name: str
    version: str
    content_hash: str
    n_tasks: int
    n_clusters: int

    @classmethod
    def from_suite(cls, suite: SuiteSpec) -> SuiteRef:
        """Build a reference from a loaded suite."""
        return cls(
            name=suite.name,
            version=suite.version,
            content_hash=suite.content_digest(),
            n_tasks=len(suite.tasks),
            n_clusters=suite.n_clusters,
        )
