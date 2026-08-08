"""Runner tests (Phase 3 acceptance).

The two claims: an interrupted run resumes to identical analysis input, and a mismatched-suite
comparison is refused rather than warned about.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agentgate.errors import ConfigError, SuiteMismatchError
from agentgate.faults import FaultConfig
from agentgate.providers.client import DEFAULT_CACHE_PATH
from agentgate.runner import (
    RunConfig,
    Runner,
    assert_comparable,
    discover_suites,
    load_suite,
    pair_trajectories,
    resolve_suite_path,
    run_suite,
    validate_suite,
)
from agentgate.schemas.common import ProviderMode
from agentgate.schemas.results import BudgetSpec
from agentgate.schemas.task import SuiteSpec
from agentgate.schemas.trajectory import RunStatus, Trajectory
from agentgate.seeds import derive_seed
from agentgate.storage.duckdb_store import RunStore
from tests.conftest import make_manifest, make_suite, make_trajectory

REPO = Path(__file__).resolve().parents[2]
SMOKE = REPO / "suites" / "smoke"


def config_for(tmp_path: Path, **overrides: object) -> RunConfig:
    defaults: dict[str, object] = {
        "suite_path": SMOKE,
        "system": "baseline",
        "mode": ProviderMode.MOCK,
        "run_root": tmp_path / "runs",
        "store_path": None,
        "cache_path": tmp_path / "cache.sqlite",
        "concurrency": 4,
    }
    defaults.update(overrides)
    return RunConfig(**defaults)  # type: ignore[arg-type]


def digests(trajectories: list[Trajectory]) -> list[str]:
    """Analysis-relevant fingerprints, free of wall-clock noise."""
    return [t.analysis_digest() for t in trajectories]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_suite_loads_from_a_directory_or_a_file() -> None:
    from_dir = load_suite(SMOKE)
    from_file = load_suite(SMOKE / "suite.yaml")
    assert from_dir == from_file
    assert from_dir.name == "smoke"
    assert len(from_dir.tasks) == 8


def test_resolve_names_the_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no suite file in"):
        resolve_suite_path(tmp_path)
    with pytest.raises(ConfigError, match="does not exist"):
        resolve_suite_path(tmp_path / "nope")


def test_ambiguous_directory_is_refused(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("name: a", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("name: b", encoding="utf-8")
    with pytest.raises(ConfigError, match="several YAML files"):
        resolve_suite_path(tmp_path)


def test_malformed_yaml_names_the_file(tmp_path: Path) -> None:
    bad = tmp_path / "suite.yaml"
    bad.write_text("name: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="malformed YAML"):
        load_suite(bad)


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "suite.yaml"
    bad.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        load_suite(bad)


def test_schema_violations_are_located(tmp_path: Path) -> None:
    bad = tmp_path / "suite.yaml"
    bad.write_text("name: bad\ntasks: []\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="tasks"):
        load_suite(bad)


def test_discovery_finds_the_shipped_suites() -> None:
    found = discover_suites(REPO / "suites")
    assert "smoke" in found
    assert found["smoke"].name == "suite.yaml"


def test_discovery_of_a_missing_root_is_empty(tmp_path: Path) -> None:
    assert discover_suites(tmp_path / "absent") == {}


def test_shipped_smoke_suite_has_no_validation_warnings() -> None:
    assert validate_suite(load_suite(SMOKE)) == []


def test_validation_warns_about_tiny_suites() -> None:
    warnings = validate_suite(make_suite(3, clusters=3))
    assert any("only 3 tasks" in warning for warning in warnings)


def test_validation_warns_about_singleton_clusters() -> None:
    tasks = load_suite(SMOKE).tasks
    merged = tasks[6].model_copy(update={"cluster_id": tasks[0].cluster_id})
    suite = SuiteSpec(name="single", tasks=[*tasks[:6], merged])
    warnings = validate_suite(suite)
    assert any("single task" in warning for warning in warnings)


def test_validation_warns_about_duplicate_prompts() -> None:
    suite = make_suite(2, clusters=1)
    duplicated = suite.model_copy(
        update={
            "tasks": [
                suite.tasks[0],
                suite.tasks[1].model_copy(update={"inputs": suite.tasks[0].inputs}),
            ]
        }
    )
    assert any("identical prompt" in warning for warning in validate_suite(duplicated))


# ---------------------------------------------------------------------------
# Seeds and identity
# ---------------------------------------------------------------------------


def test_every_unit_gets_a_distinct_deterministic_seed(tmp_path: Path) -> None:
    runner = Runner(config_for(tmp_path))
    units = runner.units()
    assert len(units) == 16
    assert len({unit.seed for unit in units}) == 16
    for unit in units:
        assert unit.seed == derive_seed(runner.config.base_seed, unit.task.id, unit.rep)


def test_run_id_is_stable_and_sensitive_to_what_matters(tmp_path: Path) -> None:
    base = Runner(config_for(tmp_path)).run_id
    assert base == Runner(config_for(tmp_path)).run_id
    assert base.startswith("smoke-baseline-")

    assert Runner(config_for(tmp_path, k=4)).run_id != base
    assert Runner(config_for(tmp_path, base_seed=1)).run_id != base
    assert Runner(config_for(tmp_path, faults=FaultConfig(verbosity=True))).run_id != base


def test_system_label_separates_otherwise_identical_runs(tmp_path: Path) -> None:
    """A no-op comparison still needs two distinct, resumable runs."""
    baseline = Runner(config_for(tmp_path, system="baseline"))
    candidate = Runner(config_for(tmp_path, system="candidate"))
    assert baseline.run_id != candidate.run_id
    assert baseline.run_id.split("-")[-1] == candidate.run_id.split("-")[-1]


def test_explicit_run_id_wins(tmp_path: Path) -> None:
    assert Runner(config_for(tmp_path, run_id="my-run")).run_id == "my-run"


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


async def test_run_produces_n_times_k_samples(tmp_path: Path) -> None:
    result = await Runner(config_for(tmp_path)).run()
    assert result.summary.n_samples == 16
    assert result.summary.status_counts == {"completed": 16}
    assert result.manifest.k == 2
    assert result.manifest.suite.n_tasks == 8
    assert [t.rep for t in result.trajectories[:2]] == [0, 1]


async def test_reps_of_the_same_task_use_different_seeds(tmp_path: Path) -> None:
    result = await Runner(config_for(tmp_path)).run()
    by_task: dict[str, set[int]] = {}
    for trajectory in result.trajectories:
        by_task.setdefault(trajectory.task_id, set()).add(trajectory.seed)
    assert all(len(seeds) == 2 for seeds in by_task.values())


async def test_results_do_not_depend_on_concurrency(tmp_path: Path) -> None:
    serial = await Runner(config_for(tmp_path / "a", concurrency=1)).run()
    parallel = await Runner(config_for(tmp_path / "b", concurrency=8)).run()
    assert digests(serial.trajectories) == digests(parallel.trajectories)


async def test_progress_hook_reports_every_unit(tmp_path: Path) -> None:
    seen: list[tuple[int, int, str]] = []
    await Runner(config_for(tmp_path)).run(progress=lambda c, t, label: seen.append((c, t, label)))
    assert len(seen) == 16
    assert seen[-1][0] == 16
    assert all(total == 16 for _, total, _ in seen)


async def test_manifest_captures_provenance(tmp_path: Path) -> None:
    result = await Runner(config_for(tmp_path, faults=FaultConfig(verbosity=True))).run()
    manifest = result.manifest
    assert manifest.faults == ["FAULT_VERBOSITY=1"]
    assert manifest.agentgate_version
    assert manifest.library_versions["pydantic"]
    assert manifest.prompt_hashes["system"]
    assert manifest.host["system"]
    assert manifest.config_hash


async def test_run_warnings_surface_suite_problems(tmp_path: Path) -> None:
    result = await Runner(config_for(tmp_path)).run()
    assert result.warnings == []


async def test_unknown_agent_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown agent"):
        await run_suite(config_for(tmp_path, agent="not_an_agent"))


# ---------------------------------------------------------------------------
# Acceptance 1: resume
# ---------------------------------------------------------------------------


async def test_interrupted_run_resumes_to_identical_analysis_input(tmp_path: Path) -> None:
    reference = await Runner(config_for(tmp_path / "full")).run()

    # Simulate an interruption: run once, then truncate the JSONL to a partial prefix.
    partial_root = tmp_path / "partial"
    first = Runner(config_for(partial_root.parent, run_root=partial_root))
    await first.run()
    jsonl = partial_root / first.run_id / "trajectories.jsonl"
    lines = jsonl.read_text(encoding="utf-8").splitlines(keepends=True)
    assert len(lines) == 16
    jsonl.write_text("".join(lines[:7]), encoding="utf-8")

    resumed = await Runner(config_for(partial_root.parent, run_root=partial_root)).run()

    assert resumed.resumed == 7
    assert any("resumed 7" in warning for warning in resumed.warnings)
    assert digests(resumed.trajectories) == digests(reference.trajectories)


async def test_resume_can_be_disabled(tmp_path: Path) -> None:
    first = Runner(config_for(tmp_path))
    await first.run()
    again = await Runner(config_for(tmp_path, resume=False)).run()
    assert again.resumed == 0


async def test_resume_is_a_noop_on_a_complete_run(tmp_path: Path) -> None:
    first = await Runner(config_for(tmp_path)).run()
    second = await Runner(config_for(tmp_path)).run()
    assert second.resumed == 16
    assert digests(second.trajectories) == digests(first.trajectories)


async def test_trajectories_are_flushed_as_they_complete(tmp_path: Path) -> None:
    runner = Runner(config_for(tmp_path))
    await runner.run()
    jsonl = tmp_path / "runs" / runner.run_id / "trajectories.jsonl"
    assert jsonl.exists()
    assert len(jsonl.read_text(encoding="utf-8").splitlines()) == 16


# ---------------------------------------------------------------------------
# Budget halt
# ---------------------------------------------------------------------------


async def test_budget_cap_halts_cleanly_and_keeps_completed_work(tmp_path: Path) -> None:
    config = config_for(
        tmp_path,
        mode=ProviderMode.LIVE,  # LIVE + the mock brain transport: budget applies to every call
        budget=BudgetSpec(max_requests=6),
        concurrency=1,
        cache_path=Path(DEFAULT_CACHE_PATH),
    )
    result = await Runner(config).run()

    statuses = {t.status for t in result.trajectories}
    assert RunStatus.BUDGET_EXHAUSTED in statuses, "unrun units are marked, not silently dropped"
    assert result.summary.n_samples == 16, "the run still accounts for every unit"
    assert any("halted early" in warning for warning in result.warnings)
    completed = [t for t in result.trajectories if t.status is RunStatus.COMPLETED]
    assert completed, "work finished before the cap is kept"


# ---------------------------------------------------------------------------
# Acceptance 2: pairing is refused, not warned about
# ---------------------------------------------------------------------------


def test_identical_configurations_are_comparable() -> None:
    suite = make_suite()
    assert_comparable(make_manifest("a", suite=suite), make_manifest("b", suite=suite))


def test_different_suites_are_refused_with_an_actionable_message() -> None:
    with pytest.raises(SuiteMismatchError, match="suite content hash differs") as excinfo:
        assert_comparable(
            make_manifest("a", suite=make_suite(4)), make_manifest("b", suite=make_suite(5))
        )
    message = str(excinfo.value)
    assert "spec C2" in message
    assert "re-run both systems" in message


@pytest.mark.parametrize(("field", "value"), [("k", 8), ("base_seed", 99)])
def test_mismatched_k_or_seed_is_refused(field: str, value: int) -> None:
    suite = make_suite()
    baseline = make_manifest("a", suite=suite)
    candidate = make_manifest("b", suite=suite).model_copy(update={field: value})
    with pytest.raises(SuiteMismatchError):
        assert_comparable(baseline, candidate)


def test_pairing_aligns_on_task_and_rep() -> None:
    baseline = [make_trajectory("t1", rep=0), make_trajectory("t2", rep=0)]
    candidate = [
        make_trajectory("t2", rep=0, system="candidate"),
        make_trajectory("t1", rep=0, system="candidate"),
    ]
    pairs = pair_trajectories(baseline, candidate)
    assert [pair.key for pair in pairs] == [("t1", 0), ("t2", 0)]
    assert all(pair.candidate.system == "candidate" for pair in pairs)


def test_pairing_refuses_unmatched_units() -> None:
    with pytest.raises(SuiteMismatchError, match="not paired"):
        pair_trajectories([make_trajectory("t1")], [make_trajectory("t2")])


def test_pairing_refuses_mismatched_seeds() -> None:
    baseline = make_trajectory("t1")
    candidate = make_trajectory("t1")
    candidate.seed = baseline.seed + 1
    with pytest.raises(SuiteMismatchError, match="different seeds"):
        pair_trajectories([baseline], [candidate])


async def test_baseline_and_candidate_runs_pair_cleanly(tmp_path: Path) -> None:
    baseline = await Runner(config_for(tmp_path, system="baseline")).run()
    candidate = await Runner(
        config_for(tmp_path, system="candidate", faults=FaultConfig(drop_tool="refund_order"))
    ).run()

    assert_comparable(baseline.manifest, candidate.manifest)
    pairs = pair_trajectories(baseline.trajectories, candidate.trajectories)
    assert len(pairs) == 16
    assert any(pair.baseline.tool_sequence != pair.candidate.tool_sequence for pair in pairs), (
        "the fault must actually change something to be worth gating on"
    )


# ---------------------------------------------------------------------------
# DuckDB persistence
# ---------------------------------------------------------------------------


async def test_run_persists_to_duckdb(tmp_path: Path) -> None:
    store_path = tmp_path / "runs.duckdb"
    result = await Runner(config_for(tmp_path, store_path=store_path)).run()

    with RunStore(store_path) as store:
        assert store.get_run(result.run_id) == result.manifest
        assert store.sample_count(result.run_id) == 16
        assert digests(store.load_trajectories(result.run_id)) == digests(result.trajectories)
        assert store.clusters_for(result.run_id)["smoke-01-refund-small"] == "refund_small"
        assert store.latest_run("smoke", "baseline") == result.run_id
        assert store.list_runs(suite="smoke")[0]["run_id"] == result.run_id


def test_store_round_trips_a_manifest_and_samples(tmp_path: Path) -> None:
    suite = make_suite()
    manifest = make_manifest("run-x", suite=suite)
    trajectories = [make_trajectory("t0", rep=rep) for rep in range(2)]
    with RunStore(tmp_path / "s.duckdb") as store:
        store.save_run(manifest)
        assert store.save_samples("run-x", trajectories, {"t0": "c0"}) == 2
        assert store.completed_units("run-x") == {("t0", 0), ("t0", 1)}
        store.delete_run("run-x")
        assert store.get_run("run-x") is None
        assert store.sample_count("run-x") == 0


def test_store_replaces_rather_than_duplicates(tmp_path: Path) -> None:
    with RunStore(tmp_path / "s.duckdb") as store:
        store.save_samples("r", [make_trajectory("t0", rep=0)], {})
        store.save_samples("r", [make_trajectory("t0", rep=0)], {})
        assert store.sample_count("r") == 1


def test_unknown_run_reads_as_none(tmp_path: Path) -> None:
    with RunStore(tmp_path / "s.duckdb") as store:
        assert store.get_run("nope") is None
        assert store.latest_run("nope", "baseline") is None
        assert store.load_trajectories("nope") == []


# ---------------------------------------------------------------------------
# Reproducibility contract
# ---------------------------------------------------------------------------


async def test_latency_is_reproducible_because_it_sums_step_durations(tmp_path: Path) -> None:
    first = await Runner(config_for(tmp_path / "a")).run()
    second = await Runner(config_for(tmp_path / "b")).run()
    assert [t.latency_ms for t in first.trajectories] == [t.latency_ms for t in second.trajectories]
    assert all(t.wall_ms > 0 for t in first.trajectories)


def test_analysis_payload_excludes_wall_clock() -> None:
    trajectory = make_trajectory()
    payload = trajectory.analysis_payload()
    assert "started_at" not in payload
    assert "ended_at" not in payload
    assert "wall_ms" not in payload
    assert payload["latency_ms"] == trajectory.latency_ms


def test_analysis_digest_ignores_wall_clock_differences() -> None:
    a = make_trajectory()
    b = make_trajectory()
    b.wall_ms = 999.0
    assert a.analysis_digest() == b.analysis_digest()
    b.final_answer = "different"
    assert a.analysis_digest() != b.analysis_digest()


def test_asyncio_run_is_not_needed_inside_the_event_loop() -> None:
    """Guard: the runner must be awaited, never nested inside a running loop."""
    assert asyncio.iscoroutinefunction(Runner.run)
