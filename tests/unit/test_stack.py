"""The published technique showcase must not claim capabilities the codebase lacks.

A portfolio page listing technologies proves nothing on its own — the claim only means something
if the code is there. These tests turn the showcase into checked assertions: every entry names a
module, and every module must import.
"""

from __future__ import annotations

import importlib

import pytest

from agentgate.report.stack import STACK, Technique, all_techniques


def test_every_claimed_technique_has_code_behind_it() -> None:
    """The load-bearing test: a name on the page must point at something that exists."""
    missing: list[str] = []
    for technique in all_techniques():
        try:
            importlib.import_module(technique.module)
        except ImportError:
            missing.append(f"{technique.name} -> {technique.module}")
    assert not missing, f"showcase claims with no implementation: {missing}"


def test_the_showcase_is_substantial_enough_to_be_worth_showing() -> None:
    assert len(STACK) >= 4
    assert len(all_techniques()) >= 20


def test_no_technique_is_listed_twice() -> None:
    names = [technique.name for technique in all_techniques()]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("technique", all_techniques(), ids=lambda t: t.name)
def test_each_entry_is_written_for_a_reader_not_a_search_engine(technique: Technique) -> None:
    assert technique.what.endswith("."), "each description is a sentence"
    assert len(technique.what) > 40, "one-word descriptions explain nothing"
    assert technique.module.startswith("agentgate.")


def test_groups_carry_their_own_framing() -> None:
    for group in STACK:
        assert group.title
        assert len(group.blurb) > 40
        assert group.items
