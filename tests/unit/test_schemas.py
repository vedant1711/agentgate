"""Schema round-trip, hashing, and validation tests (Phase 0 acceptance)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agentgate.schemas import (
    ArgComparator,
    ComparisonResult,
    Estimate,
    GatedMetric,
    GatePolicy,
    GateVerdict,
    MetricFamily,
    MetricResult,
    RunManifest,
    SuiteRef,
    SuiteSpec,
    TaskSpec,
    Trajectory,
    Verdict,
)
from tests.conftest import make_manifest, make_suite, make_task, make_trajectory

# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_suite_round_trips_through_json(suite: SuiteSpec) -> None:
    restored = SuiteSpec.model_validate_json(suite.model_dump_json())
    assert restored == suite
    assert restored.content_digest() == suite.content_digest()


def test_task_round_trips_through_json(task: TaskSpec) -> None:
    assert TaskSpec.model_validate_json(task.model_dump_json()) == task


def test_trajectory_round_trips_and_preserves_step_types(trajectory: Trajectory) -> None:
    restored = Trajectory.model_validate_json(trajectory.model_dump_json())
    assert restored == trajectory
    assert [step.kind for step in restored.steps] == [step.kind for step in trajectory.steps]


def test_manifest_round_trips_and_keeps_config_hash() -> None:
    manifest = make_manifest()
    restored = RunManifest.model_validate_json(manifest.model_dump_json())
    assert restored.config_hash == manifest.config_hash


def test_policy_round_trips_through_yaml() -> None:
    policy = GatePolicy.default()
    yaml_text = _dump_yaml(policy.model_dump(mode="json"))
    assert GatePolicy.model_validate_yaml(yaml_text) == policy


def test_gate_verdict_round_trips() -> None:
    verdict = GateVerdict(
        comparison_id="cmp-1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        verdict=Verdict.REGRESSION,
        exit_code=1,
        baseline_run_id="a",
        candidate_run_id="b",
        suite=SuiteRef.from_suite(make_suite()),
        metric_verdicts={"outcome.task_success": Verdict.REGRESSION},
        failing_metrics=["outcome.task_success"],
    )
    restored = GateVerdict.model_validate_json(verdict.model_dump_json())
    assert restored == verdict
    assert restored.passed is False


def test_comparison_result_round_trips() -> None:
    result = ComparisonResult(
        comparison_id="cmp-1",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        baseline_run_id="a",
        candidate_run_id="b",
        suite=SuiteRef.from_suite(make_suite()),
        k=4,
    )
    assert ComparisonResult.model_validate_json(result.model_dump_json()) == result


def test_metric_result_round_trips() -> None:
    result = MetricResult(
        metric="outcome.task_success",
        task_id="t1",
        rep=0,
        system="baseline",
        value=1.0,
        family=MetricFamily.OUTCOME,
        dtype="binary",
    )
    assert MetricResult.model_validate_json(result.model_dump_json()) == result
    assert result.is_scored


# ---------------------------------------------------------------------------
# Hashing determinism (A3.4)
# ---------------------------------------------------------------------------


def test_content_hash_is_insensitive_to_task_field_order() -> None:
    a = make_suite(2)
    b = SuiteSpec.model_validate(a.model_dump(mode="json"))
    assert a.content_digest() == b.content_digest()


def test_content_hash_changes_when_a_task_changes() -> None:
    a = make_suite(2)
    edited = a.model_copy(
        update={
            "tasks": [a.tasks[0].model_copy(update={"inputs": {"prompt": "different"}}), a.tasks[1]]
        }
    )
    assert edited.content_digest() != a.content_digest()


def test_config_hash_ignores_wall_clock_and_host() -> None:
    a = make_manifest()
    b = a.model_copy(
        update={
            "created_at": datetime(2030, 6, 1, tzinfo=UTC),
            "host": {"system": "Linux"},
            "run_id": "run-other",
        }
    )
    assert a.config_hash == b.config_hash


def test_config_hash_tracks_faults_and_models() -> None:
    a = make_manifest()
    assert a.model_copy(update={"faults": ["FAULT_DROP_TOOL=refund"]}).config_hash != a.config_hash
    assert a.model_copy(update={"k": 8}).config_hash != a.config_hash


# ---------------------------------------------------------------------------
# Comparability guard (C2)
# ---------------------------------------------------------------------------


def test_runs_on_identical_suites_are_comparable() -> None:
    suite = make_suite()
    ok, reason = make_manifest("a", suite=suite).is_comparable_to(make_manifest("b", suite=suite))
    assert ok
    assert reason == ""


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        ({"k": 2}, "K differs"),
        ({"base_seed": 999}, "base seed differs"),
    ],
)
def test_mismatched_runs_are_refused(kwargs: dict[str, int], needle: str) -> None:
    suite = make_suite()
    baseline = make_manifest("a", suite=suite)
    candidate = make_manifest("b", suite=suite, **kwargs)  # type: ignore[arg-type]
    ok, reason = baseline.is_comparable_to(candidate)
    assert not ok
    assert needle in reason


def test_different_suite_content_is_refused() -> None:
    ok, reason = make_manifest("a", suite=make_suite(4)).is_comparable_to(
        make_manifest("b", suite=make_suite(5))
    )
    assert not ok
    assert "suite content hash differs" in reason


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------


def test_duplicate_task_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate task id"):
        SuiteSpec(name="dup", tasks=[make_task("t1"), make_task("t1")])


def test_task_without_a_prompt_is_rejected() -> None:
    with pytest.raises(ValidationError, match="prompt/question/instruction"):
        TaskSpec(id="t1", cluster_id="c1", inputs={"payload": "no prompt here"})


def test_regex_comparator_requires_a_pattern() -> None:
    with pytest.raises(ValidationError, match="requires a `pattern`"):
        ArgComparator(kind="regex")


def test_gated_metric_requires_exactly_one_margin() -> None:
    with pytest.raises(ValidationError, match="exactly one of margin"):
        GatedMetric(metric="outcome.task_success")
    with pytest.raises(ValidationError, match="exactly one of margin"):
        GatedMetric(metric="outcome.task_success", margin=0.03, margin_ratio=1.2)


def test_duplicate_gated_metrics_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate gated metric"):
        GatePolicy(
            gated_metrics=[
                GatedMetric(metric="trajectory.f1", margin=0.03),
                GatedMetric(metric="trajectory.f1", margin=0.05),
            ]
        )


def test_reliability_metrics_are_keyed_by_k() -> None:
    entry = GatedMetric(metric="reliability.pass_hat_k", k=4, margin=0.05)
    assert entry.resolved_name == "reliability.pass_hat_k@4"
    assert entry.margin_kind == "absolute"


def test_extra_fields_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        TaskSpec(id="t1", cluster_id="c1", inputs={"prompt": "hi"}, bogus=1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Derived views
# ---------------------------------------------------------------------------


def test_tool_invocations_pair_calls_with_results(trajectory: Trajectory) -> None:
    calls = trajectory.tool_invocations
    assert [c.tool for c in calls] == ["lookup_customer", "refund"]
    assert all(c.ok for c in calls)
    assert trajectory.tool_sequence == ["lookup_customer", "refund"]
    assert trajectory.n_tool_calls == 2
    assert trajectory.n_llm_roundtrips == 2


def test_orphan_tool_call_is_reported_as_failed() -> None:
    traj = make_trajectory()
    traj.steps = [step for step in traj.steps if step.kind != "tool_result"]
    calls = traj.tool_invocations
    assert len(calls) == 2
    assert all(not c.ok for c in calls)
    assert all(c.error == "no tool result recorded" for c in calls)


def test_failed_tool_result_is_visible(trajectory: Trajectory) -> None:
    failed = make_trajectory(failing_call_index=0)
    assert [c.ok for c in failed.tool_invocations] == [False, True]
    assert [c.ok for c in trajectory.tool_invocations] == [True, True]


def test_invocation_signature_is_order_insensitive_over_args() -> None:
    a = make_trajectory(tools=("refund",), args=({"order_id": 42, "amount": 10},))
    b = make_trajectory(tools=("refund",), args=({"amount": 10, "order_id": 42},))
    assert a.tool_invocations[0].signature == b.tool_invocations[0].signature


def test_usage_accumulates_across_llm_calls(trajectory: Trajectory) -> None:
    assert trajectory.usage.prompt_tokens == 200
    assert trajectory.usage.completion_tokens == 40
    assert trajectory.usage.total_tokens == 240


def test_evidence_bank_collects_successful_tool_outputs(trajectory: Trajectory) -> None:
    assert trajectory.evidence_bank == ["lookup_customer ok", "refund ok"]


def test_suite_cluster_helpers() -> None:
    clustered = make_suite(6, clusters=2)
    assert clustered.n_clusters == 2
    assert clustered.has_clusters
    unclustered = make_suite(3, clusters=3)
    assert not unclustered.has_clusters
    assert clustered.task("t0").id == "t0"
    with pytest.raises(KeyError, match="not in suite"):
        clustered.task("missing")


def test_reference_trajectory_counts_required_steps() -> None:
    task = make_task()
    assert task.reference.trajectory is not None
    assert task.reference.trajectory.required_step_count == 2


def test_estimate_half_width() -> None:
    est = Estimate(value=0.5, se=0.05, ci_low=0.4, ci_high=0.6, method="clt", n=30)
    assert est.half_width == pytest.approx(0.1)
    assert Estimate(value=0.5, method="degenerate", n=1).half_width is None


def _dump_yaml(payload: object) -> str:
    import yaml

    return yaml.safe_dump(payload, sort_keys=True)
