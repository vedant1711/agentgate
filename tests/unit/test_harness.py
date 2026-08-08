"""Living-harness tests: the ledger, the scheduler, the leaderboard, and trends.

The one claim worth defending above all others here: **the leaderboard must never assert an
ordering the evidence does not support.** A board that ranks 1, 2, 3 by point estimate would
commit, inside the harness, precisely the error the gate exists to prevent in CI. So most of these
tests are about refusing to claim things.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agentgate.errors import InsufficientDataError
from agentgate.harness import (
    Cell,
    Ledger,
    LedgerEntry,
    assign_tiers,
    build_leaderboard,
    build_trend,
    estimate_seconds,
    head_to_head,
    model_of,
    plan,
)
from agentgate.harness.leaderboard import Standing
from agentgate.harness.schedule import UNKNOWN_COST
from agentgate.harness.trend import INCOMPARABLE, REGRESSED, STABLE
from agentgate.providers.models import get_card
from agentgate.schemas import (
    Estimate,
    MetricFamily,
    MetricResult,
    ModelRef,
    RunManifest,
    SuiteRef,
    SuiteSpec,
)
from agentgate.storage.duckdb_store import RunStore
from tests.conftest import FIXED_TIME, make_manifest, make_suite, make_trajectory

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def manifest_for(
    run_id: str,
    *,
    model_id: str,
    suite: SuiteSpec,
    k: int = 2,
    created_at: datetime = FIXED_TIME,
) -> RunManifest:
    """A manifest that names its agent model, which is what the ledger keys on."""
    base = make_manifest(run_id, suite=suite, k=k)
    return base.model_copy(
        update={
            "created_at": created_at,
            "models": [ModelRef(role="agent", model_id=model_id, provider="ollama_chat")],
        }
    )


def scores_for(
    run_id: str, task_values: dict[str, float], *, k: int = 2, metric: str = "outcome.task_success"
) -> list[MetricResult]:
    """Per-sample results giving each task a constant value across its repetitions."""
    return [
        MetricResult(
            metric=metric,
            task_id=task_id,
            rep=rep,
            system="baseline",
            value=value,
            family=MetricFamily.OUTCOME,
            dtype="binary",
        )
        for task_id, value in task_values.items()
        for rep in range(k)
    ]


def seed_run(
    store: RunStore,
    *,
    run_id: str,
    model_id: str,
    suite: SuiteSpec,
    values: dict[str, float],
    k: int = 2,
    created_at: datetime = FIXED_TIME,
    score: bool = True,
) -> None:
    """Write a complete, scored run for one model into the store."""
    store.save_run(manifest_for(run_id, model_id=model_id, suite=suite, k=k, created_at=created_at))
    trajectories = [make_trajectory(task_id, rep=rep) for task_id in values for rep in range(k)]
    store.save_samples(run_id, trajectories, {task.id: task.cluster_id for task in suite.tasks})
    if score:
        store.save_scores(run_id, scores_for(run_id, values, k=k))


@pytest.fixture
def suite() -> SuiteSpec:
    return make_suite()


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "harness.duckdb")


def standing(model_id: str, value: float, low: float, high: float, tier: int = 1) -> Standing:
    return Standing(
        model_id=model_id,
        label=model_id,
        run_id=f"run-{model_id}",
        recorded_at=FIXED_TIME,
        estimate=Estimate(value=value, ci_low=low, ci_high=high, method="clt", n=10),
        n_tasks=10,
        k=2,
        completion_rate=1.0,
        tier=tier,
    )


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def test_the_ledger_is_rebuilt_from_runs_not_from_bookkeeping(
    store: RunStore, suite: SuiteSpec
) -> None:
    """No second source of truth means the ledger cannot drift from the runs it describes."""
    values = {task.id: 1.0 for task in suite.tasks}
    seed_run(store, run_id="r1", model_id="ollama_chat/llama3.2:3b", suite=suite, values=values)

    ledger = Ledger.from_store(store)
    assert ledger.models() == ["ollama_chat/llama3.2:3b"]
    assert ledger.suites() == [suite.name]
    entry = ledger.latest(Cell(suite=suite.name, model_id="ollama_chat/llama3.2:3b", k=2))
    assert entry is not None
    assert entry.is_complete
    assert entry.is_scored


def test_a_run_that_does_not_name_its_model_is_not_attributed_to_one(
    store: RunStore, suite: SuiteSpec
) -> None:
    """Guessing would put another model's score on the wrong row."""
    manifest = make_manifest("r1", suite=suite, k=2).model_copy(update={"models": []})
    assert model_of(manifest) == "unknown"
    store.save_run(manifest)
    assert Ledger.from_store(store).models() == ["unknown"]


def test_a_partial_recording_is_reported_as_incomplete(store: RunStore, suite: SuiteSpec) -> None:
    values = {task.id: 1.0 for task in suite.tasks}
    seed_run(store, run_id="r1", model_id="m", suite=suite, values=values, k=2)
    store._conn.execute("DELETE FROM samples WHERE run_id = 'r1' AND rep = 1")

    entry = Ledger.from_store(store).latest(Cell(suite=suite.name, model_id="m", k=2))
    assert entry is not None
    assert not entry.is_complete
    assert entry.n_samples < entry.expected_samples


def test_missing_returns_both_never_recorded_and_partial_cells(
    store: RunStore, suite: SuiteSpec
) -> None:
    values = {task.id: 1.0 for task in suite.tasks}
    seed_run(store, run_id="r1", model_id="done", suite=suite, values=values, k=2)

    missing = Ledger.from_store(store).missing(suites=[suite.name], models=["done", "never"], k=2)
    assert [cell.model_id for cell in missing] == ["never"]


def test_a_recorded_but_unscored_cell_still_counts_as_missing(
    store: RunStore, suite: SuiteSpec
) -> None:
    """An unscored recording cannot enter a leaderboard, so the work is not finished."""
    values = {task.id: 1.0 for task in suite.tasks}
    seed_run(store, run_id="r1", model_id="m", suite=suite, values=values, score=False)

    missing = Ledger.from_store(store).missing(suites=[suite.name], models=["m"], k=2)
    assert [cell.model_id for cell in missing] == ["m"]


def test_completion_rate_is_about_the_model_not_the_harness() -> None:
    entry = LedgerEntry(
        cell=Cell(suite="s", model_id="m", k=2),
        run_id="r",
        recorded_at=FIXED_TIME,
        agent="a",
        system="baseline",
        n_tasks=10,
        n_samples=20,
        n_completed=15,
        n_scores=20,
        git_sha="0" * 40,
    )
    assert entry.is_complete
    assert entry.completion_rate == 0.75


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def test_a_model_with_no_measured_rate_has_no_invented_cost() -> None:
    assert estimate_seconds(None, n_tasks=10, k=3) == UNKNOWN_COST
    assert estimate_seconds(get_card("ollama_chat/qwen2.5:7b"), n_tasks=10, k=3) == UNKNOWN_COST


def test_cost_estimates_scale_with_tasks_and_repetitions() -> None:
    card = get_card("ollama_chat/llama3.2:3b")
    assert card is not None
    single = estimate_seconds(card, n_tasks=10, k=1)
    assert estimate_seconds(card, n_tasks=10, k=3) == pytest.approx(single * 3)
    assert estimate_seconds(card, n_tasks=20, k=1) == pytest.approx(single * 2)


def test_breadth_beats_depth_in_the_queue(store: RunStore, suite: SuiteSpec) -> None:
    """An unmeasured model carries unbounded uncertainty; a partial one carries merely wide."""
    values = {task.id: 1.0 for task in suite.tasks}
    seed_run(store, run_id="r1", model_id="ollama_chat/llama3.2:3b", suite=suite, values=values)
    store._conn.execute("DELETE FROM samples WHERE run_id = 'r1' AND rep = 1")

    queued = plan(
        Ledger.from_store(store),
        suites=[suite.name],
        models=["ollama_chat/llama3.2:3b", "mock/agent"],
        k=2,
        budget_seconds=0,
        sizes={suite.name: len(suite.tasks)},
    )
    assert [cell.cell.model_id for cell in queued.cells] == [
        "mock/agent",
        "ollama_chat/llama3.2:3b",
    ]
    assert queued.cells[1].resumable


def test_unknown_cost_cells_are_deferred_so_one_slow_model_cannot_eat_a_session(
    store: RunStore, suite: SuiteSpec
) -> None:
    queued = plan(
        Ledger([]),
        suites=[suite.name],
        models=["ollama_chat/qwen2.5:7b", "mock/agent"],
        k=1,
        budget_seconds=3600,
        sizes={suite.name: len(suite.tasks)},
    )
    assert [cell.cell.model_id for cell in queued.cells] == ["mock/agent"]
    assert [cell.cell.model_id for cell in queued.deferred] == ["ollama_chat/qwen2.5:7b"]


def test_work_that_does_not_fit_the_budget_is_deferred_not_dropped(suite: SuiteSpec) -> None:
    """Silently dropping work would make the next session's plan wrong."""
    queued = plan(
        Ledger([]),
        suites=[suite.name],
        models=["ollama_chat/llama3.2:3b", "mock/agent"],
        k=1,
        budget_seconds=1.0,
        sizes={suite.name: 1000},
    )
    total = len(queued.cells) + len(queued.deferred)
    assert total == 2
    assert queued.deferred


def test_a_suite_that_cannot_be_sized_is_skipped_rather_than_guessed(suite: SuiteSpec) -> None:
    queued = plan(
        Ledger([]),
        suites=["nonexistent"],
        models=["mock/agent"],
        k=1,
        budget_seconds=0,
        sizes={},
    )
    assert queued.cells == ()


# ---------------------------------------------------------------------------
# Tiering — the core claim
# ---------------------------------------------------------------------------


def test_models_with_overlapping_intervals_share_a_tier() -> None:
    tiers = assign_tiers([standing("a", 0.60, 0.45, 0.75), standing("b", 0.55, 0.40, 0.70)])
    assert [item.tier for item in tiers] == [1, 1]


def test_models_with_disjoint_intervals_are_separated() -> None:
    tiers = assign_tiers([standing("a", 0.90, 0.85, 0.95), standing("b", 0.20, 0.10, 0.30)])
    assert [item.tier for item in tiers] == [1, 2]


def test_tiering_compares_against_the_tier_leader_not_the_predecessor() -> None:
    """The transitivity trap, which is what makes naive tiering worse than none.

    Here a overlaps b and b overlaps c, but a and c are disjoint. Chaining each model to its
    predecessor would put all three in one tier — asserting a and c are indistinguishable when
    their intervals do not touch. Comparing against the tier *leader* splits c off correctly.
    """
    tiers = assign_tiers(
        [
            standing("a", 0.90, 0.80, 1.00),
            standing("b", 0.78, 0.70, 0.85),
            standing("c", 0.63, 0.55, 0.72),
        ]
    )
    assert [item.tier for item in tiers] == [1, 1, 2]  # predecessor-chaining would give [1, 1, 1]


def test_a_model_without_a_computable_interval_gets_its_own_tier() -> None:
    """Nothing can be claimed about it, so it must not be absorbed into a claim about others."""
    unknown = Standing(
        model_id="u",
        label="u",
        run_id="r",
        recorded_at=FIXED_TIME,
        estimate=Estimate(value=0.5, method="degenerate", n=1),
        n_tasks=1,
        k=1,
        completion_rate=1.0,
    )
    tiers = assign_tiers([standing("a", 0.90, 0.80, 1.00), unknown])
    assert [item.tier for item in tiers] == [1, 2]


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def test_the_leaderboard_ranks_models_it_can_separate(store: RunStore, suite: SuiteSpec) -> None:
    strong = {task.id: 1.0 for task in suite.tasks}
    weak = {task.id: 0.0 for task in suite.tasks}
    seed_run(store, run_id="r-strong", model_id="strong", suite=suite, values=strong)
    seed_run(store, run_id="r-weak", model_id="weak", suite=suite, values=weak)

    board = build_leaderboard(store, suite=suite.name)
    assert [item.model_id for item in board.standings] == ["strong", "weak"]
    assert board.is_separable
    assert "Tier 1" in board.verdict()


def test_the_leaderboard_refuses_to_rank_models_it_cannot_separate(
    store: RunStore, suite: SuiteSpec
) -> None:
    """Two models scoring identically must come back tied, not 1 and 2."""
    values = {task.id: 1.0 if index % 2 else 0.0 for index, task in enumerate(suite.tasks)}
    seed_run(store, run_id="r-a", model_id="a", suite=suite, values=values)
    seed_run(store, run_id="r-b", model_id="b", suite=suite, values=values)

    board = build_leaderboard(store, suite=suite.name)
    assert len(board.standings) == 2
    assert not board.is_separable
    assert {item.tier for item in board.standings} == {1}
    assert "none separable" in board.verdict()
    assert "head-to-head" in board.verdict()


def test_partial_recordings_are_excluded_with_a_stated_reason(
    store: RunStore, suite: SuiteSpec
) -> None:
    """Ranking 2-of-4 tasks beside 4-of-4 implies a comparison that was never made."""
    values = {task.id: 1.0 for task in suite.tasks}
    seed_run(store, run_id="r-full", model_id="full", suite=suite, values=values)
    seed_run(store, run_id="r-part", model_id="part", suite=suite, values=values)
    store._conn.execute("DELETE FROM samples WHERE run_id = 'r-part' AND rep = 1")

    board = build_leaderboard(store, suite=suite.name)
    assert [item.model_id for item in board.standings] == ["full"]
    assert dict(board.excluded)["part"].startswith("partial recording")


def test_an_unscored_run_is_excluded_rather_than_scored_as_zero(
    store: RunStore, suite: SuiteSpec
) -> None:
    values = {task.id: 1.0 for task in suite.tasks}
    seed_run(store, run_id="r", model_id="m", suite=suite, values=values, score=False)

    board = build_leaderboard(store, suite=suite.name)
    assert board.standings == ()
    assert dict(board.excluded)["m"] == "recorded but never scored"


def test_a_one_model_board_says_so_rather_than_declaring_a_winner(
    store: RunStore, suite: SuiteSpec
) -> None:
    values = {task.id: 1.0 for task in suite.tasks}
    seed_run(store, run_id="r", model_id="only", suite=suite, values=values)
    assert "at least two" in build_leaderboard(store, suite=suite.name).verdict()


def test_an_empty_board_does_not_pretend_to_have_measured_anything(store: RunStore) -> None:
    assert "No model" in build_leaderboard(store, suite="nothing").verdict()


# ---------------------------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------------------------


def test_head_to_head_pairs_two_models_on_the_tasks_they_both_ran(
    store: RunStore, suite: SuiteSpec
) -> None:
    strong = {task.id: 1.0 for task in suite.tasks}
    weak = {task.id: 0.0 for task in suite.tasks}
    seed_run(store, run_id="r-a", model_id="a", suite=suite, values=weak)
    seed_run(store, run_id="r-b", model_id="b", suite=suite, values=strong)

    comparison = head_to_head(store, suite=suite.name, baseline_model="a", candidate_model="b")
    assert comparison.metric == "outcome.task_success"
    assert comparison.delta.value > 0


def test_head_to_head_names_exactly_what_is_missing(store: RunStore, suite: SuiteSpec) -> None:
    values = {task.id: 1.0 for task in suite.tasks}
    seed_run(store, run_id="r-a", model_id="a", suite=suite, values=values)

    with pytest.raises(InsufficientDataError, match="'ghost' has no run"):
        head_to_head(store, suite=suite.name, baseline_model="a", candidate_model="ghost")


# ---------------------------------------------------------------------------
# Trends
# ---------------------------------------------------------------------------


def test_a_trend_needs_two_points_before_it_says_anything(
    store: RunStore, suite: SuiteSpec
) -> None:
    values = {task.id: 1.0 for task in suite.tasks}
    seed_run(store, run_id="r1", model_id="m", suite=suite, values=values)

    trend = build_trend(
        store, cell=Cell(suite=suite.name, model_id="m", k=2), metric="outcome.task_success"
    )
    assert len(trend.points) == 1
    assert "at least two" in trend.describe()


def test_overlapping_intervals_are_reported_as_noise_not_movement(
    store: RunStore, suite: SuiteSpec
) -> None:
    """Two points differing slightly, with wide intervals, establish nothing."""
    first = {task.id: 1.0 if index % 2 else 0.0 for index, task in enumerate(suite.tasks)}
    second = dict(first)
    second[suite.tasks[0].id] = 1.0

    seed_run(store, run_id="r1", model_id="m", suite=suite, values=first, created_at=FIXED_TIME)
    seed_run(
        store,
        run_id="r2",
        model_id="m",
        suite=suite,
        values=second,
        created_at=FIXED_TIME + timedelta(days=1),
    )

    trend = build_trend(
        store, cell=Cell(suite=suite.name, model_id="m", k=2), metric="outcome.task_success"
    )
    assert [move.verdict for move in trend.movements] == [STABLE]
    assert trend.established_moves == []
    assert "no established movement" in trend.describe()


def test_a_genuine_collapse_is_reported_as_a_regression(store: RunStore, suite: SuiteSpec) -> None:
    perfect = {task.id: 1.0 for task in suite.tasks}
    zero = {task.id: 0.0 for task in suite.tasks}
    seed_run(store, run_id="r1", model_id="m", suite=suite, values=perfect, created_at=FIXED_TIME)
    seed_run(
        store,
        run_id="r2",
        model_id="m",
        suite=suite,
        values=zero,
        created_at=FIXED_TIME + timedelta(days=1),
    )

    trend = build_trend(
        store, cell=Cell(suite=suite.name, model_id="m", k=2), metric="outcome.task_success"
    )
    assert [move.verdict for move in trend.movements] == [REGRESSED]
    assert trend.movements[0].delta < 0


def test_a_changed_suite_breaks_the_trend_instead_of_drawing_through_it(
    store: RunStore, suite: SuiteSpec
) -> None:
    """Scores from different suites answer different questions; a line between them is a lie."""
    values = {task.id: 1.0 for task in suite.tasks}
    seed_run(store, run_id="r1", model_id="m", suite=suite, values=values, created_at=FIXED_TIME)

    later = manifest_for(
        "r2",
        model_id="m",
        suite=suite,
        created_at=FIXED_TIME + timedelta(days=1),
    )
    moved = SuiteRef.from_suite(suite).model_copy(update={"content_hash": "f" * 64})
    store.save_run(later.model_copy(update={"suite": moved}))
    store.save_samples(
        "r2",
        [make_trajectory(task.id, rep=rep) for task in suite.tasks for rep in range(2)],
        {task.id: task.cluster_id for task in suite.tasks},
    )
    store.save_scores("r2", scores_for("r2", {task.id: 0.0 for task in suite.tasks}))

    trend = build_trend(
        store, cell=Cell(suite=suite.name, model_id="m", k=2), metric="outcome.task_success"
    )
    assert [move.verdict for move in trend.movements] == [INCOMPARABLE]
    assert "different questions" in trend.movements[0].describe()


def test_trend_points_are_ordered_oldest_first(store: RunStore, suite: SuiteSpec) -> None:
    values = {task.id: 1.0 for task in suite.tasks}
    seed_run(store, run_id="r1", model_id="m", suite=suite, values=values, created_at=FIXED_TIME)
    seed_run(
        store,
        run_id="r2",
        model_id="m",
        suite=suite,
        values=values,
        created_at=FIXED_TIME + timedelta(days=1),
    )
    trend = build_trend(
        store, cell=Cell(suite=suite.name, model_id="m", k=2), metric="outcome.task_success"
    )
    assert [point.run_id for point in trend.points] == ["r1", "r2"]
    assert trend.points[0].recorded_at < trend.points[1].recorded_at


def test_now_is_not_used_anywhere_in_a_trend(store: RunStore, suite: SuiteSpec) -> None:
    """Timestamps come from manifests, so a trend is reproducible from the store alone."""
    values = {task.id: 1.0 for task in suite.tasks}
    seed_run(store, run_id="r1", model_id="m", suite=suite, values=values, created_at=FIXED_TIME)
    trend = build_trend(
        store, cell=Cell(suite=suite.name, model_id="m", k=2), metric="outcome.task_success"
    )
    assert trend.points[0].recorded_at == FIXED_TIME
    assert trend.points[0].recorded_at < datetime.now(UTC)
