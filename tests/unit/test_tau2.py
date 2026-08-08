"""tau2-bench retail adaptation: sandbox rules, suite conversion, and the trajectory checker.

The claim these tests defend is narrow but important: the tau2 suite is only worth adopting if
its *business rules are enforced by the sandbox*. If the sandbox let an agent cancel a delivered
order, then "policy compliance" would be a matter of opinion rather than an observable tool error,
and every downstream statistic would be measuring the checker's mood instead of the model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agentgate.agents.tau2_retail import (
    DEFAULT_DB_PATH,
    RETAIL_TOOLS,
    Tau2RetailSandbox,
    load_retail_db,
)
from agentgate.faults.config import FaultConfig
from agentgate.metrics.base import ScoredSample
from agentgate.metrics.checkers import CHECKERS, check_trajectory_reference
from agentgate.schemas.task import (
    CheckerSpec,
    ReferenceSpec,
    ReferenceStep,
    ReferenceTrajectory,
    TaskSpec,
)
from agentgate.schemas.trajectory import ToolCallStep, ToolResultStep, Trajectory

DATA_PRESENT = DEFAULT_DB_PATH.exists()
needs_data = pytest.mark.skipif(not DATA_PRESENT, reason="tau2 retail data not vendored")

SUITE_PATH = Path(__file__).resolve().parents[2] / "suites" / "tau2_retail" / "suite.yaml"


@pytest.fixture
def box() -> Tau2RetailSandbox:
    return Tau2RetailSandbox()


def _pending_order(db: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for order_id, order in db["orders"].items():
        if order["status"] == "pending":
            return order_id, order
    msg = "no pending order in the vendored database"
    raise AssertionError(msg)


def _delivered_order(db: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for order_id, order in db["orders"].items():
        if order["status"] == "delivered":
            return order_id, order
    msg = "no delivered order in the vendored database"
    raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------


def test_every_declared_tool_has_a_handler(box: Tau2RetailSandbox) -> None:
    """A declared-but-unhandled tool would read as a model failure, not a harness bug."""
    for spec in RETAIL_TOOLS:
        outcome = box.invoke(spec.name, {})
        assert outcome.error != f"unknown tool {spec.name!r}", spec.name


@needs_data
def test_unknown_tools_are_errors_not_exceptions(box: Tau2RetailSandbox) -> None:
    outcome = box.invoke("delete_everything", {})
    assert not outcome.ok
    assert "unknown tool" in (outcome.error or "")


@needs_data
def test_bad_arguments_come_back_as_tool_errors(box: Tau2RetailSandbox) -> None:
    """A model that calls a tool wrong must see an error it can react to, not crash the run."""
    outcome = box.invoke("get_order_details", {})
    assert not outcome.ok
    assert "bad arguments" in (outcome.error or "")


@needs_data
def test_dropping_a_tool_hides_it_and_rejects_it(box: Tau2RetailSandbox) -> None:
    hidden = Tau2RetailSandbox(faults=FaultConfig(drop_tool="get_order_details"))
    assert "get_order_details" not in hidden.tool_names
    assert not hidden.invoke("get_order_details", {"order_id": "#W0000000"}).ok


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


@needs_data
def test_each_sandbox_starts_from_an_untouched_world() -> None:
    """One task's refund must not silently change the next task's answer."""
    first = Tau2RetailSandbox()
    order_id, _ = _pending_order(first._db)
    assert first.invoke(
        "cancel_pending_order", {"order_id": order_id, "reason": "no longer needed"}
    ).ok

    second = Tau2RetailSandbox()
    assert second._db["orders"][order_id]["status"] == "pending"


@needs_data
def test_snapshot_reports_only_what_changed() -> None:
    box = Tau2RetailSandbox()
    assert box.snapshot() == {"orders": {}, "users": {}}

    order_id, _ = _pending_order(box._db)
    box.invoke("cancel_pending_order", {"order_id": order_id, "reason": "ordered by mistake"})
    snapshot = box.snapshot()
    assert set(snapshot["orders"]) == {order_id}
    assert snapshot["orders"][order_id]["status"] == "cancelled"


# ---------------------------------------------------------------------------
# The domain's actual policies
# ---------------------------------------------------------------------------


@needs_data
def test_only_pending_orders_can_be_cancelled(box: Tau2RetailSandbox) -> None:
    order_id, _ = _delivered_order(box._db)
    outcome = box.invoke(
        "cancel_pending_order", {"order_id": order_id, "reason": "no longer needed"}
    )
    assert not outcome.ok
    assert box._db["orders"][order_id]["status"] == "delivered"


@needs_data
def test_cancellation_reason_is_restricted_to_the_two_allowed_strings(
    box: Tau2RetailSandbox,
) -> None:
    """tau2 accepts exactly two reasons. Accepting more would make the policy unmeasurable."""
    order_id, _ = _pending_order(box._db)
    assert not box.invoke("cancel_pending_order", {"order_id": order_id, "reason": "why not"}).ok
    assert box._db["orders"][order_id]["status"] == "pending"


@needs_data
def test_only_pending_orders_can_have_their_address_modified(box: Tau2RetailSandbox) -> None:
    order_id, _ = _delivered_order(box._db)
    outcome = box.invoke(
        "modify_pending_order_address",
        {
            "order_id": order_id,
            "address1": "1 New St",
            "address2": "",
            "city": "Springfield",
            "state": "CA",
            "country": "USA",
            "zip": "99999",
        },
    )
    assert not outcome.ok


@needs_data
def test_lookup_by_email_is_case_insensitive_and_misses_cleanly(box: Tau2RetailSandbox) -> None:
    user_id, user = next(iter(box._db["users"].items()))
    found = box.invoke("find_user_id_by_email", {"email": str(user["email"]).upper()})
    assert found.ok
    assert found.output == user_id
    assert not box.invoke("find_user_id_by_email", {"email": "nobody@example.com"}).ok


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_a_missing_database_names_the_command_that_fixes_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build_tau2_suite"):
        load_retail_db(tmp_path / "absent.json")


# ---------------------------------------------------------------------------
# The trajectory_reference checker
# ---------------------------------------------------------------------------


def _sample(gold: list[str], called: list[str]) -> ScoredSample:
    task = TaskSpec(
        id="t",
        cluster_id="c",
        inputs={"prompt": "do it"},
        checker=CheckerSpec(name="trajectory_reference"),
        reference=ReferenceSpec(
            trajectory=ReferenceTrajectory(
                required_tools=sorted(set(gold)),
                allow_extra_calls=True,
                steps=[ReferenceStep(tools=[tool]) for tool in gold],
            )
        ),
    )
    trajectory = Trajectory(task_id="t", rep=0, seed=0, system="baseline")
    for index, tool in enumerate(called):
        call_id = f"c{index}"
        trajectory.steps.append(
            ToolCallStep(index=len(trajectory.steps), call_id=call_id, tool=tool, args={})
        )
        trajectory.steps.append(
            ToolResultStep(index=len(trajectory.steps), call_id=call_id, tool=tool, ok=True)
        )
    return ScoredSample(task=task, trajectory=trajectory)


def test_the_checker_is_registered_under_the_name_the_suite_uses() -> None:
    assert CHECKERS["trajectory_reference"] is check_trajectory_reference


def test_the_trajectory_checker_passes_only_when_every_gold_call_was_made() -> None:
    gold = ["get_order_details", "cancel_pending_order"]
    passed, _ = check_trajectory_reference(_sample(gold, gold), {})
    assert passed

    missing, detail = check_trajectory_reference(_sample(gold, ["get_order_details"]), {})
    assert not missing
    assert detail["missing"] == ["cancel_pending_order"]


def test_the_trajectory_checker_tolerates_extra_calls() -> None:
    """A model that looks something up twice has not failed the task."""
    passed, _ = check_trajectory_reference(
        _sample(
            ["get_order_details"],
            ["list_all_product_types", "get_order_details", "calculate"],
        ),
        {},
    )
    assert passed


def test_the_trajectory_checker_requires_the_gold_order_by_default() -> None:
    """tau2's gold sequences are causal — you cannot cancel an order you have not looked up."""
    gold = ["get_order_details", "cancel_pending_order"]
    reversed_calls = ["cancel_pending_order", "get_order_details"]
    strict, _ = check_trajectory_reference(_sample(gold, reversed_calls), {})
    assert not strict

    relaxed, _ = check_trajectory_reference(_sample(gold, reversed_calls), {"in_order": False})
    assert relaxed


def test_a_task_without_a_reference_trajectory_fails_loudly() -> None:
    task = TaskSpec(id="t", cluster_id="c", inputs={"prompt": "do it"})
    trajectory = Trajectory(task_id="t", rep=0, seed=0, system="baseline")
    passed, detail = check_trajectory_reference(ScoredSample(task=task, trajectory=trajectory), {})
    assert not passed
    assert "no reference trajectory" in detail["reason"]


# ---------------------------------------------------------------------------
# The generated suite
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SUITE_PATH.exists(), reason="tau2 suite not generated")
def test_the_generated_suite_keeps_tau2_ground_truth_and_says_it_is_an_adaptation() -> None:
    import yaml

    document = yaml.safe_load(SUITE_PATH.read_text(encoding="utf-8"))
    assert document["metadata"]["source_license"] == "MIT"
    assert "NOT comparable" in document["description"]

    tasks = document["tasks"]
    assert len(tasks) > 100, "the whole point of adopting tau2 was sample size"
    # Every task its own cluster: tau2 tasks are independent, and inventing clusters would
    # overstate correlation and shrink the effective sample size for no reason.
    assert len({task["cluster_id"] for task in tasks}) == len(tasks)
    for task in tasks:
        assert task["reference"]["trajectory"]["required_tools"]


@needs_data
def test_every_gold_tool_in_the_suite_exists_in_the_sandbox() -> None:
    """If the suite demanded a tool the sandbox lacks, no model could ever pass that task."""
    import yaml

    if not SUITE_PATH.exists():
        pytest.skip("tau2 suite not generated")
    document = yaml.safe_load(SUITE_PATH.read_text(encoding="utf-8"))
    available = {spec.name for spec in RETAIL_TOOLS}
    demanded = {
        tool
        for task in document["tasks"]
        for tool in task["reference"]["trajectory"]["required_tools"]
    }
    assert demanded <= available, demanded - available


@needs_data
def test_the_vendored_database_is_the_shape_the_sandbox_expects() -> None:
    db = json.loads(DEFAULT_DB_PATH.read_text(encoding="utf-8"))
    assert set(db) >= {"users", "orders", "products"}
    assert len(db["orders"]) > 100
