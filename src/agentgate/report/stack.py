"""The evaluation stack, as a structured claim rather than a logo wall.

A portfolio page that lists technologies proves nothing — anyone can name a library. What is
worth showing is *which technique solves which problem*, because that is the part a reader cannot
fake and the part an interviewer will probe.

So every entry here names the technique, states in one line what it is for, and points at the
module that implements it. If a claim on the published page has no implementation behind it, the
test in ``tests/unit/test_stack.py`` fails — which makes this file a set of checked claims rather
than marketing copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class Technique:
    """One capability the project implements.

    Args:
        name: What it is called in the literature or in the ecosystem.
        what: One line on what it does, written for someone who has not read the docs.
        module: Import path of the code that implements it — the receipt for the claim.
        external: True when it is a third-party technology rather than something built here.
    """

    name: str
    what: str
    module: str
    external: bool = False


@dataclass(frozen=True, slots=True)
class StackGroup:
    """A themed group of techniques."""

    title: str
    blurb: str
    items: tuple[Technique, ...]


STACK: Final[tuple[StackGroup, ...]] = (
    StackGroup(
        title="Agent evaluation",
        blurb=(
            "Grading what the agent <em>did</em>, not only what it said. Final-answer scoring "
            "cannot tell a correct answer from a lucky one."
        ),
        items=(
            Technique(
                "Trajectory evaluation",
                "Scores the tool calls themselves — selection, ordering, argument correctness, "
                "redundancy, loop detection, recovery after an error.",
                "agentgate.metrics.trajectory",
            ),
            Technique(
                "τ²-bench",
                "111 tasks from Sierra's published agent benchmark, graded against their own "
                "gold trajectories in a 16-tool environment.",
                "agentgate.agents.tau2_retail",
                external=True,
            ),
            Technique(
                "pass^k reliability",
                "The τ-bench metric that exposes agents whose average looks fine and whose "
                "consistency does not. Unbiased combinatorial estimator.",
                "agentgate.stats.reliability",
            ),
            Technique(
                "ReAct tool-calling agents",
                "Reason–act loops over a sandboxed company: 10 tools, enforced business rules, "
                "so a policy violation is an observable error rather than an opinion.",
                "agentgate.agents.tool_agent",
            ),
            Technique(
                "Fault injection",
                "Eight knobs that break the agent through the system — prompts genuinely "
                "deleted, tools genuinely removed — so the gate is tested against real damage.",
                "agentgate.faults.config",
            ),
        ),
    ),
    StackGroup(
        title="LLM-as-a-judge, controlled",
        blurb=(
            "A judge is an instrument with error and bias. Treating its score as ground truth is "
            "the most common mistake in modern evaluation."
        ),
        items=(
            Technique(
                "Position-swap averaging",
                "Judges prefer whichever answer came first. Every comparison is graded both ways "
                "and averaged.",
                "agentgate.judge.rubric_judge",
            ),
            Technique(
                "Verbosity & markdown audits",
                "Spearman correlation between score and answer length, so a judge rewarding "
                "wordiness is caught rather than trusted.",
                "agentgate.judge.audits",
            ),
            Technique(
                "Judge/agent family independence",
                "Default-deny when judge and agent share a model family — self-preference is "
                "measurable and disqualifying.",
                "agentgate.judge.independence",
            ),
            Technique(
                "Cohen's κ calibration floor",
                "The judge is checked against human labels and refuses to be used below κ 0.6.",
                "agentgate.judge.calibration",
            ),
            Technique(
                "Frozen-anchor drift detection",
                "A pinned anchor set is re-graded every run, so a provider silently updating the "
                "model underneath you is detected.",
                "agentgate.judge.drift",
            ),
            Technique(
                "Judge variance propagation",
                "The judge's own variance is folded into the metric's standard error through the "
                "law of total variance.",
                "agentgate.stats.variance",
            ),
        ),
    ),
    StackGroup(
        title="Statistical inference",
        blurb=(
            "The part that turns a score difference into a decision. Every method here is chosen "
            "for a specific failure of the naive alternative."
        ),
        items=(
            Technique(
                "Paired non-inferiority testing",
                "Asks whether the agent got worse by more than a declared margin — not whether "
                "anything changed, because something always changes.",
                "agentgate.stats.paired",
            ),
            Technique(
                "CR1 cluster-robust errors",
                "Five rewordings of one scenario count as one fact. Ignoring that makes the gate "
                "fire on noise.",
                "agentgate.stats.intervals",
            ),
            Technique(
                "Benjamini–Hochberg FDR",
                "Testing 40 metrics at α=0.05 false-alarms ~87% of the time by chance. BH "
                "controls the discovery rate across the gated family.",
                "agentgate.stats.multiplicity",
            ),
            Technique(
                "BCa bootstrap",
                "Bias-corrected accelerated intervals, resampled at the cluster level, for "
                "statistics with no closed form.",
                "agentgate.stats.intervals",
            ),
            Technique(
                "Power & minimum detectable effect",
                "Reports what the suite <em>could</em> have caught, so a pass is never "
                "mistaken for evidence of absence.",
                "agentgate.stats.power",
            ),
            Technique(
                "Permutation tests",
                "Used wherever the parametric assumption fails — heavy ties, non-zero margins, "
                "small clusters.",
                "agentgate.stats.paired",
            ),
        ),
    ),
    StackGroup(
        title="Observability & interop",
        blurb="Standard instrumentation, so this plugs into what teams already run.",
        items=(
            Technique(
                "OpenTelemetry + OpenInference",
                "Agent, LLM, tool and retriever spans under the semantic conventions the LLM "
                "observability ecosystem standardised on.",
                "agentgate.tracing",
                external=True,
            ),
            Technique(
                "Arize Phoenix",
                "Self-hosted trace UI for debugging a failing task, via OTLP. Optional by "
                "design — the trajectory, not the span, is the source of truth.",
                "agentgate.tracing",
                external=True,
            ),
            Technique(
                "LiteLLM routing",
                "One interface across Ollama, Groq, Gemini and NVIDIA NIM, with per-provider "
                "reachability checks before a run starts.",
                "agentgate.providers.litellm_transport",
                external=True,
            ),
            Technique(
                "RAGAS / DeepEval export",
                "Recorded runs export into either framework's input shape, with the "
                "definitional differences documented rather than glossed.",
                "agentgate.interop",
                external=True,
            ),
        ),
    ),
    StackGroup(
        title="Systems & reproducibility",
        blurb=(
            "A verdict you cannot reproduce is an opinion. The infrastructure exists to make the "
            "same inputs give the same answer, byte for byte."
        ),
        items=(
            Technique(
                "Record / replay provider layer",
                "Four modes. CI runs in replay, where a cache miss is a hard error — so the "
                "gate never calls a model and never costs anything.",
                "agentgate.providers.client",
            ),
            Technique(
                "Content-hashed suites & config",
                "A changed suite can never be silently compared against an old baseline.",
                "agentgate.schemas.common",
            ),
            Technique(
                "DuckDB per-sample store",
                "Scores kept per sample, never aggregated, so any run can be re-analysed at a "
                "different K, margin, or alpha without re-running the agent.",
                "agentgate.storage.duckdb_store",
                external=True,
            ),
            Technique(
                "pydantic v2 contracts",
                "Every boundary typed and exported as JSON Schema, with "
                "<code>mypy --strict</code> across the whole codebase.",
                "agentgate.schemas",
                external=True,
            ),
            Technique(
                "Property-based testing",
                "Hypothesis generates the adversarial inputs — the invariants hold or the build "
                "fails.",
                "agentgate.stats",
                external=True,
            ),
        ),
    ),
)


def all_techniques() -> list[Technique]:
    """Flatten the stack into a single list."""
    return [item for group in STACK for item in group.items]
