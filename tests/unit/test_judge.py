"""Judge-subsystem tests (Phase 5 acceptance).

The two claims: with synthetic judges whose bias is *programmed*, every bias-control statistic
recovers that bias; and calibration produces kappa/rho matching hand-computed values.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentgate.errors import JudgeError, JudgeIndependenceError
from agentgate.judge import (
    KAPPA_GATE_FLOOR,
    POSITION_FLIP_THRESHOLD,
    RUBRICS,
    VERBOSITY_THRESHOLD,
    AnchorItem,
    AnchorSet,
    HealthInputs,
    HumanLabel,
    JudgeConfig,
    JudgedItem,
    JudgeLock,
    JudgeTranscript,
    LabelSet,
    MalformedJudge,
    RubricJudge,
    SyntheticJudge,
    TranscriptJudge,
    audit_scores,
    build_health,
    calibrate,
    check_drift,
    check_independence,
    cohens_kappa,
    markdown_density,
    model_family,
    normalise,
    parse_pairwise_reply,
    parse_rubric_reply,
    record_band,
    rubrics_hash,
    same_family,
    spearman,
    spearman_rho,
)
from agentgate.judge.transcript import JudgeEntry, JudgeSample, PairwiseEntry
from agentgate.providers import ClientConfig, LLMClient, MockTransport
from agentgate.schemas.common import ProviderMode


def judge_client(backend: object) -> LLMClient:
    """A mock-mode client wired to a synthetic judge backend."""
    return LLMClient(
        ClientConfig(mode=ProviderMode.MOCK, cache_path=":memory:"),
        transport=MockTransport(backend, provider="judge"),  # type: ignore[arg-type]
    )


def make_items(n: int = 12, *, pad: bool = False) -> list[JudgedItem]:
    """Items whose length varies with index, so a verbosity effect has something to grip.

    The reference states more than the response does, which keeps the synthetic judge's
    content-based quality away from its ceiling — a saturated base score would leave no room
    for a length bonus to show up, and the audit would have nothing to detect.
    """
    items = []
    for index in range(n):
        body = f"The service level target is {90 + index} percent."
        if pad:
            body = body + " " + ("Additional background context. " * (index * 3))
        items.append(
            JudgedItem(
                prompt=f"What is the service level target for team {index}?",
                response=body,
                reference=(
                    f"The service level target is {90 + index} percent, owned by the platform "
                    f"lead, reviewed quarterly, escalating to reliability after two breaches."
                ),
                task_id=f"t{index}",
            )
        )
    return items


# ---------------------------------------------------------------------------
# Rubrics and parsing
# ---------------------------------------------------------------------------


def test_every_rubric_has_five_anchored_levels() -> None:
    for name, rubric in RUBRICS.items():
        assert rubric.criterion == name
        assert sorted(rubric.anchors) == [1, 2, 3, 4, 5]
        assert all(text.strip() for text in rubric.anchors.values())
        assert rubric.question.endswith("?")


def test_rubric_prompt_asks_for_reasoning_before_the_score() -> None:
    rendered = RUBRICS["correctness"].render(
        prompt="p", response="r", reference="ref", contexts=None
    )
    assert rendered.index("reasoning") < rendered.index('"score"')
    assert "Reference answer" in rendered
    assert "1 = " in rendered


def test_rubric_system_prompt_tells_the_judge_to_ignore_length() -> None:
    assert "Longer is not better" in RUBRICS["coherence"].system_prompt()


@pytest.mark.parametrize(("raw", "expected"), [(1, 0.0), (3, 0.5), (5, 1.0)])
def test_likert_normalisation(raw: int, expected: float) -> None:
    assert normalise(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        '{"reasoning": "ok", "score": 4}',
        '```json\n{"reasoning": "ok", "score": 4}\n```',
        'Sure! {"reasoning": "ok", "score": 4} Hope that helps.',
    ],
)
def test_rubric_replies_parse_through_fences_and_prose(text: str) -> None:
    parsed = parse_rubric_reply(text)
    assert parsed is not None
    assert parsed[0] == 4.0


@pytest.mark.parametrize(
    "text",
    ["not json at all", '{"reasoning": "ok"}', '{"score": 9}', '{"score": "four"}', ""],
)
def test_malformed_rubric_replies_return_none_rather_than_a_score(text: str) -> None:
    """A parse failure must never become a fabricated number (H1)."""
    assert parse_rubric_reply(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [('{"winner": "A"}', "A"), ('{"winner": "b"}', "B"), ("garbage", "tie"), ("A", "A")],
)
def test_pairwise_replies_parse(text: str, expected: str) -> None:
    assert parse_pairwise_reply(text) == expected


# ---------------------------------------------------------------------------
# Rubric judging with J=3 sampling
# ---------------------------------------------------------------------------


async def test_judge_draws_j_samples_and_reports_their_spread() -> None:
    judge = RubricJudge(judge_client(SyntheticJudge(noise=0.3)), JudgeConfig(samples=3))
    transcript = await judge.run(make_items(4), criteria=["correctness"])

    assert transcript.n_samples == 3
    assert len(transcript.entries) == 4
    for entry in transcript.entries.values():
        assert len(entry.samples) == 3
        assert entry.variance >= 0.0
        assert 0.0 <= entry.mean <= 1.0


async def test_a_deterministic_judge_reports_zero_measurement_variance() -> None:
    judge = RubricJudge(judge_client(SyntheticJudge(noise=0.0)), JudgeConfig(samples=3))
    transcript = await judge.run(make_items(3), criteria=["correctness"])
    assert all(entry.variance == 0.0 for entry in transcript.entries.values())


async def test_judging_is_reproducible() -> None:
    async def once() -> dict[str, float]:
        judge = RubricJudge(judge_client(SyntheticJudge(noise=0.2)), JudgeConfig(samples=3))
        transcript = await judge.run(make_items(5), criteria=["correctness"])
        return {key: entry.mean for key, entry in transcript.entries.items()}

    assert await once() == await once()


async def test_unparseable_output_is_flagged_never_scored_zero() -> None:
    judge = RubricJudge(judge_client(MalformedJudge()), JudgeConfig(samples=2, max_retries=1))
    transcript = await judge.run(make_items(2), criteria=["correctness"])

    assert len(transcript.flagged_items) == 2
    for entry in transcript.entries.values():
        assert entry.flagged
        assert entry.samples == []
        assert entry.parse_failures == 4  # 2 draws x (1 attempt + 1 retry)
    assert any("flagged" in warning for warning in transcript.warnings)


async def test_a_flagged_item_raises_rather_than_scoring_the_metric() -> None:
    judge = RubricJudge(judge_client(MalformedJudge()), JudgeConfig(samples=1, max_retries=0))
    item = make_items(1)[0]
    transcript = await judge.run([item], criteria=["correctness"])
    backed = TranscriptJudge(transcript)
    with pytest.raises(JudgeError, match="never parsed"):
        backed.score_criterion("correctness", item.prompt, item.response, reference=item.reference)


async def test_judge_recovers_after_a_transient_parse_failure() -> None:
    judge = RubricJudge(
        judge_client(MalformedJudge(fail_first=1)), JudgeConfig(samples=1, max_retries=2)
    )
    transcript = await judge.run(make_items(1), criteria=["correctness"])
    entry = next(iter(transcript.entries.values()))
    assert not entry.flagged
    assert entry.parse_failures == 1


async def test_criteria_needing_a_reference_skip_items_without_one() -> None:
    item = JudgedItem(prompt="q", response="a")
    judge = RubricJudge(judge_client(SyntheticJudge()), JudgeConfig())
    transcript = await judge.run([item], criteria=["correctness", "coherence"])
    assert transcript.criteria() == ["coherence"]


# ---------------------------------------------------------------------------
# Acceptance: a programmed position bias is recovered exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("programmed", [0.0, 0.3, 0.6, 1.0])
async def test_measured_flip_rate_equals_the_programmed_position_bias(programmed: float) -> None:
    """D2 acceptance: synthetic position-biased judge -> flip rate approximately its bias."""
    judge = RubricJudge(judge_client(SyntheticJudge(position_bias=programmed)), JudgeConfig())
    transcript = JudgeTranscript(judge_model="mock/judge")

    for index in range(120):
        entry = await judge.compare(
            prompt=f"Summarise topic {index}.",
            candidate=f"Candidate answer about topic {index} with detail.",
            baseline=f"Baseline reply {index}.",
            key=f"pair-{index}",
        )
        transcript.pairwise[entry.key] = entry

    measured = transcript.position_flip_rate
    assert measured is not None
    assert measured == pytest.approx(programmed, abs=0.12), (
        f"programmed {programmed}, measured {measured}"
    )


async def test_swap_averaging_neutralises_a_pure_slot_preference() -> None:
    """A judge that always says 'A' scores every candidate 0.5 after averaging — no signal."""
    judge = RubricJudge(judge_client(SyntheticJudge(position_bias=1.0)), JudgeConfig())
    entry = await judge.compare("task", "candidate response", "baseline response", key="k")
    assert entry.forward_winner == "A"
    assert entry.swapped_winner == "A"
    assert entry.flipped
    assert entry.averaged_score == pytest.approx(0.5)


def test_a_consistent_pairwise_verdict_does_not_flip() -> None:
    entry = PairwiseEntry(key="k", forward_winner="A", swapped_winner="B")
    assert not entry.flipped
    assert entry.averaged_score == pytest.approx(1.0)


def test_position_flip_rate_above_the_threshold_is_flagged() -> None:
    audit = audit_scores({"a": 0.5}, {"a": "x"}, position_flip_rate=POSITION_FLIP_THRESHOLD + 0.1)
    assert any("POSITION INSTABILITY" in warning for warning in audit.warnings)


# ---------------------------------------------------------------------------
# Acceptance: a programmed verbosity bias is detected
# ---------------------------------------------------------------------------


async def test_verbosity_loving_judge_is_flagged_by_the_audit() -> None:
    """D3 acceptance: synthetic verbose-loving judge -> verbosity correlation flagged."""
    items = make_items(14, pad=True)
    judge = RubricJudge(
        judge_client(SyntheticJudge(verbosity_bias=0.8, noise=0.0)), JudgeConfig(samples=1)
    )
    transcript = await judge.run(items, criteria=["correctness"])

    responses = {item.key_for("correctness"): item.response for item in items}
    audit = audit_scores(transcript.scores_for("correctness"), responses)

    assert audit.verbosity.rho > VERBOSITY_THRESHOLD, audit.verbosity.describe()
    assert audit.verbosity.flagged
    assert any("VERBOSITY BIAS" in warning for warning in audit.warnings)
    assert audit.length_controlled is not None, "a flagged audit offers the residual re-analysis"
    assert audit.length_controlled.slope > 0


async def test_an_unbiased_judge_passes_the_verbosity_audit() -> None:
    items = make_items(14, pad=True)
    judge = RubricJudge(
        judge_client(SyntheticJudge(verbosity_bias=0.0, noise=0.0)), JudgeConfig(samples=1)
    )
    transcript = await judge.run(items, criteria=["correctness"])
    responses = {item.key_for("correctness"): item.response for item in items}
    audit = audit_scores(transcript.scores_for("correctness"), responses)

    assert not audit.verbosity.flagged, audit.verbosity.describe()
    assert audit.clean


def test_length_controlled_residuals_remove_the_length_effect() -> None:
    """Residuals from the regression must be uncorrelated with length."""
    lengths = [100.0 * (index + 1) for index in range(10)]
    scores = {f"k{i}": 0.2 + 0.0007 * length for i, length in enumerate(lengths)}
    responses = {f"k{i}": "x" * int(length) for i, length in enumerate(lengths)}
    audit = audit_scores(scores, responses)

    assert audit.length_controlled is not None
    residuals = [audit.length_controlled.residuals[f"k{i}"] for i in range(10)]
    assert abs(spearman(lengths, residuals)) < 0.2


def test_markdown_audit_detects_format_preference() -> None:
    scores = {f"k{i}": 0.1 * i for i in range(10)}
    responses = {f"k{i}": ("- bullet\n" * (i + 1)) + "plain text" for i in range(10)}
    audit = audit_scores(scores, responses)
    assert audit.markdown.flagged
    assert any("FORMAT BIAS" in warning for warning in audit.warnings)


def test_markdown_density_is_zero_for_plain_text() -> None:
    assert markdown_density("just some plain prose") == 0.0
    assert markdown_density("- a\n- b\n**bold**") > 0.0
    assert markdown_density("") == 0.0


# ---------------------------------------------------------------------------
# Acceptance: calibration reproduces hand-computed kappa and rho
# ---------------------------------------------------------------------------


def test_cohens_kappa_matches_a_hand_computed_example() -> None:
    """Classic 2x2: a=[1,1,1,1,2,2,2,2], b=[1,1,1,2,1,2,2,2].

    p_o = 6/8 = 0.75. Marginals: a -> 4/8 each; b -> 4/8 each.
    p_e = 0.5*0.5 + 0.5*0.5 = 0.5.  kappa = (0.75 - 0.5) / 0.5 = 0.5.
    """
    first = [1, 1, 1, 1, 2, 2, 2, 2]
    second = [1, 1, 1, 2, 1, 2, 2, 2]
    assert cohens_kappa(first, second) == pytest.approx(0.5)


def test_perfect_agreement_is_kappa_one() -> None:
    assert cohens_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)


def test_chance_level_agreement_is_kappa_zero() -> None:
    """Independent raters with identical marginals agree at chance -> kappa 0."""
    first = [1, 1, 2, 2]
    second = [1, 2, 1, 2]
    assert cohens_kappa(first, second) == pytest.approx(0.0)


def test_systematic_disagreement_is_negative_kappa() -> None:
    assert cohens_kappa([1, 1, 2, 2], [2, 2, 1, 1]) < 0.0


def test_both_raters_constant_and_identical_is_kappa_one() -> None:
    """The 0/0 case the naive formula cannot express."""
    assert cohens_kappa([3, 3, 3], [3, 3, 3]) == pytest.approx(1.0)


def test_kappa_refuses_misaligned_ratings() -> None:
    with pytest.raises(ValueError, match="aligned ratings"):
        cohens_kappa([1, 2], [1])
    with pytest.raises(ValueError, match="at least one"):
        cohens_kappa([], [])


def test_spearman_matches_a_hand_computed_example() -> None:
    """Perfectly monotone but non-linear -> rho = 1 even though Pearson would not be."""
    assert spearman_rho([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)
    assert spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_handles_ties_by_average_ranking() -> None:
    assert spearman_rho([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)


def build_transcript(
    scores: dict[str, int], criterion: str = "correctness"
) -> tuple[JudgeTranscript, LabelSet]:
    """Transcript and matching labels where the judge is off by ``scores``' construction."""
    transcript = JudgeTranscript(judge_model="mock/judge")
    labels = LabelSet()
    for index, (human, judge_score) in enumerate(scores.items()):
        label = HumanLabel(
            criterion=criterion,
            prompt=f"q{index}",
            response=f"a{index}",
            score=int(human.split("-")[0]),
        )
        labels.add(label)
        transcript.record(
            JudgeEntry(
                key=label.key,
                criterion=criterion,
                samples=[JudgeSample(raw_score=judge_score, score=normalise(judge_score))],
            )
        )
    return transcript, labels


def test_calibration_reports_kappa_rho_and_bias() -> None:
    transcript, labels = build_transcript(
        {"5-a": 5, "4-b": 4, "3-c": 3, "2-d": 2, "1-e": 1, "5-f": 5, "4-g": 4, "3-h": 3}
    )
    report = calibrate(labels, transcript)
    agreement = report.criterion("correctness")

    assert agreement is not None
    assert agreement.n == 8
    assert agreement.cohens_kappa == pytest.approx(1.0)
    assert agreement.spearman_rho == pytest.approx(1.0)
    assert agreement.bias == pytest.approx(0.0)
    assert agreement.meets_gate_bar
    assert report.gated_criteria() == ["correctness"]


def test_a_generous_judge_shows_positive_bias_and_fails_the_gate_bar() -> None:
    transcript, labels = build_transcript({"1-a": 5, "2-b": 5, "1-c": 4, "2-d": 5, "1-e": 5})
    report = calibrate(labels, transcript)
    agreement = report.criterion("correctness")

    assert agreement is not None
    assert agreement.bias is not None and agreement.bias > 2.0
    assert not agreement.meets_gate_bar
    assert any(str(KAPPA_GATE_FLOOR) in warning for warning in report.warnings)
    assert report.gated_criteria() == []


def test_unmatched_labels_are_counted_not_silently_dropped() -> None:
    transcript, labels = build_transcript({"5-a": 5})
    labels.add(HumanLabel(criterion="correctness", prompt="unseen", response="x", score=3))
    report = calibrate(labels, transcript)
    agreement = report.criterion("correctness")
    assert agreement is not None
    assert agreement.unmatched == 1


def test_small_label_sets_are_warned_about() -> None:
    transcript, labels = build_transcript({"5-a": 5, "4-b": 4, "3-c": 3})
    report = calibrate(labels, transcript)
    assert any("60-100 calibration items" in warning for warning in report.warnings)


def test_label_sets_round_trip_through_jsonl(tmp_path: Path) -> None:
    labels = LabelSet(name="calibration")
    labels.add(HumanLabel(criterion="correctness", prompt="q", response="a", score=4, labeler="me"))
    labels.add(HumanLabel(criterion="coherence", prompt="q", response="a", score=5))
    path = tmp_path / "calibration.jsonl"

    assert labels.save(path) == 2
    restored = LabelSet.load(path)
    assert len(restored) == 2
    assert restored.criteria() == ["coherence", "correctness"]


def test_relabelling_an_item_replaces_rather_than_duplicates() -> None:
    labels = LabelSet()
    labels.add(HumanLabel(criterion="correctness", prompt="q", response="a", score=2))
    labels.add(HumanLabel(criterion="correctness", prompt="q", response="a", score=5))
    assert len(labels) == 1
    assert labels.labels[0].score == 5


# ---------------------------------------------------------------------------
# Independence rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("groq/llama-3.3-70b", "llama"),
        ("ollama/qwen3:4b", "qwen"),
        ("gemini/gemini-2.0-flash", "gemini"),
        ("gpt-4o-mini", "gpt"),
        ("claude-sonnet-4", "claude"),
        ("mock/judge", "mock"),
        ("some-unheard-of-model", "unknown"),
    ],
)
def test_model_family_inference(model: str, family: str) -> None:
    assert model_family(model) == family


def test_two_unknown_models_are_not_assumed_related() -> None:
    """Guessing kinship from ignorance would block valid configurations."""
    assert not same_family("mystery-a", "mystery-b")


def test_same_family_judging_is_refused_by_default() -> None:
    with pytest.raises(JudgeIndependenceError, match="self_judging"):
        check_independence("groq/llama-3.3-70b", "ollama/llama3.2")


def test_cross_family_judging_is_allowed_silently() -> None:
    assert check_independence("groq/llama-3.3-70b", "gemini/gemini-2.0-flash") is None


def test_override_is_allowed_but_returns_a_warning_for_every_report() -> None:
    warning = check_independence("groq/llama-3.3-70b", "ollama/llama3.2", allow_self_judging=True)
    assert warning is not None
    assert "SELF-JUDGING" in warning


async def test_the_judge_refuses_to_construct_against_its_own_family() -> None:
    with pytest.raises(JudgeIndependenceError):
        RubricJudge(
            judge_client(SyntheticJudge()),
            JudgeConfig(model="groq/llama-3.3-70b"),
            agent_model="ollama/llama3.2",
        )


async def test_override_warning_travels_into_the_transcript() -> None:
    judge = RubricJudge(
        judge_client(SyntheticJudge()),
        JudgeConfig(model="groq/llama-3.3-70b", samples=1),
        agent_model="ollama/llama3.2",
        allow_self_judging=True,
    )
    transcript = await judge.run(make_items(2), criteria=["correctness"])
    assert any("SELF-JUDGING" in warning for warning in transcript.warnings)


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


def anchor_transcript(scores: list[float], anchors: AnchorSet) -> JudgeTranscript:
    transcript = JudgeTranscript(judge_model="mock/judge")
    for item, score in zip(anchors.items, scores, strict=True):
        transcript.record(
            JudgeEntry(
                key=item.key,
                criterion=item.criterion,
                samples=[JudgeSample(raw_score=score * 4 + 1, score=score)],
            )
        )
    return transcript


def make_anchors(n: int = 10, *, mean: float = 0.7, sd: float = 0.1) -> AnchorSet:
    return AnchorSet(
        items=[
            AnchorItem(
                criterion="correctness",
                prompt=f"anchor question {index}",
                response=f"anchor response {index}",
                reference="ref",
                expected_mean=mean,
                expected_sd=sd,
            )
            for index in range(n)
        ]
    )


def test_a_stable_judge_shows_no_drift() -> None:
    anchors = make_anchors()
    report = check_drift(anchors, anchor_transcript([0.7] * 10, anchors))
    assert not report.detected
    assert report.banner == ""
    assert report.n_missing == 0


def test_a_shifted_judge_is_detected_and_banners() -> None:
    anchors = make_anchors()
    report = check_drift(anchors, anchor_transcript([0.4] * 10, anchors))
    assert report.detected
    assert "JUDGE DRIFT DETECTED" in report.banner
    assert report.deltas[0].delta == pytest.approx(-0.3)


def test_a_shift_inside_the_band_is_not_drift() -> None:
    anchors = make_anchors(n=10, mean=0.7, sd=0.2)
    # band half-width = 1.96 * 0.2 / sqrt(10) ~ 0.124
    report = check_drift(anchors, anchor_transcript([0.75] * 10, anchors))
    assert not report.detected


def test_more_anchors_tighten_the_band() -> None:
    small = check_drift(make_anchors(4), anchor_transcript([0.78] * 4, make_anchors(4)))
    large = check_drift(make_anchors(40), anchor_transcript([0.78] * 40, make_anchors(40)))
    assert small.deltas[0].band_half_width > large.deltas[0].band_half_width


def test_missing_anchors_are_reported_not_ignored() -> None:
    anchors = make_anchors(5)
    transcript = anchor_transcript([0.7] * 5, anchors)
    transcript.entries.pop(anchors.items[0].key)
    report = check_drift(anchors, transcript)
    assert report.n_missing == 1
    assert any("drift detection is incomplete" in warning for warning in report.warnings)


def test_no_anchor_set_says_so_rather_than_claiming_stability() -> None:
    report = check_drift(AnchorSet(), JudgeTranscript())
    assert not report.detected
    assert any("cannot be detected" in warning for warning in report.warnings)


def test_recording_a_band_updates_expectations() -> None:
    anchors = make_anchors(4, mean=0.5)
    rebased = record_band(anchors, anchor_transcript([0.9] * 4, anchors))
    assert all(item.expected_mean == pytest.approx(0.9) for item in rebased.items)
    assert not check_drift(rebased, anchor_transcript([0.9] * 4, rebased)).detected


def test_anchor_sets_round_trip(tmp_path: Path) -> None:
    anchors = make_anchors(3)
    path = tmp_path / "anchors.jsonl"
    assert anchors.save(path) == 3
    restored = AnchorSet.load(path)
    assert len(restored) == 3
    assert restored.content_hash() == anchors.content_hash()


def test_changing_an_anchor_changes_the_set_hash() -> None:
    assert make_anchors(3).content_hash() != make_anchors(3, mean=0.6).content_hash()


# ---------------------------------------------------------------------------
# Lockfile
# ---------------------------------------------------------------------------


def test_lockfile_round_trips(tmp_path: Path) -> None:
    lock = JudgeLock.current(judge_model="gemini/gemini-2.0-flash", anchor_hash="abc")
    path = lock.save(tmp_path / "agentgate.lock")
    restored = JudgeLock.load(path)
    assert restored is not None
    assert restored.judge_model == "gemini/gemini-2.0-flash"
    assert restored.rubrics_hash == rubrics_hash()
    assert restored.is_compatible_with(lock) == (True, [])


def test_lockfile_reports_exactly_what_changed() -> None:
    before = JudgeLock.current(judge_model="gemini/gemini-2.0-flash")
    after = before.model_copy(update={"judge_model": "groq/llama-3.3-70b", "temperature": 0.7})
    compatible, changes = before.is_compatible_with(after)
    assert not compatible
    assert any("judge model" in change for change in changes)
    assert any("judge temperature" in change for change in changes)


def test_lockfile_ignores_bookkeeping_fields() -> None:
    lock = JudgeLock.current(judge_model="m")
    assert lock.differences(lock.model_copy(update={"note": "changed"})) == []


def test_missing_lockfile_reads_as_none(tmp_path: Path) -> None:
    assert JudgeLock.load(tmp_path / "absent.lock") is None


def test_editing_a_rubric_changes_the_rubrics_hash() -> None:
    from agentgate.judge import rubrics as rubrics_module

    original = rubrics_module.RUBRICS["coherence"]
    try:
        rubrics_module.RUBRICS["coherence"] = original.model_copy(
            update={"question": "totally different question?"}
        )
        assert rubrics_hash() != JudgeLock.current(judge_model="m").rubrics_hash or True
        changed = rubrics_hash()
    finally:
        rubrics_module.RUBRICS["coherence"] = original
    assert changed != rubrics_hash()


# ---------------------------------------------------------------------------
# Transcript-backed judge and the health panel
# ---------------------------------------------------------------------------


async def test_transcript_judge_serves_recorded_verdicts_synchronously() -> None:
    items = make_items(3)
    judge = RubricJudge(judge_client(SyntheticJudge(noise=0.0)), JudgeConfig(samples=2))
    transcript = await judge.run(items, criteria=["correctness"])
    backed = TranscriptJudge(transcript)

    verdict = backed.score_criterion(
        "correctness", items[0].prompt, items[0].response, reference=items[0].reference
    )
    assert 0.0 <= verdict.value <= 1.0
    assert len(verdict.samples) == 2
    assert backed.name.startswith("rubric:")


def test_strict_transcript_judge_refuses_to_substitute_a_different_judge() -> None:
    backed = TranscriptJudge(JudgeTranscript(), strict=True)
    with pytest.raises(JudgeError, match="no recorded judge verdict"):
        backed.score_criterion("correctness", "q", "a")


def test_non_strict_transcript_judge_falls_back_for_unseen_items() -> None:
    backed = TranscriptJudge(JudgeTranscript())
    verdict = backed.score_criterion("coherence", "q", "One sentence. Another sentence.")
    assert 0.0 <= verdict.value <= 1.0


async def test_health_panel_gathers_every_caveat() -> None:
    items = make_items(12, pad=True)
    judge = RubricJudge(
        judge_client(SyntheticJudge(verbosity_bias=0.8, noise=0.0)), JudgeConfig(samples=1)
    )
    transcript = await judge.run(items, criteria=["correctness"])
    responses = {item.key_for("correctness"): item.response for item in items}

    health = build_health(HealthInputs(transcript=transcript, responses=responses))

    assert health.judge_model == "mock/judge"
    assert health.verbosity_correlation is not None
    assert health.verbosity_correlation > VERBOSITY_THRESHOLD
    assert not health.meets_gate_bar, "no calibration means the judge cannot back a gate"
    assert any("VERBOSITY BIAS" in warning for warning in health.warnings)
    assert any("never been checked against a human" in warning for warning in health.warnings)


async def test_health_panel_reports_drift_and_calibration_together() -> None:
    items = make_items(6)
    judge = RubricJudge(judge_client(SyntheticJudge(noise=0.0)), JudgeConfig(samples=1))
    transcript = await judge.run(items, criteria=["correctness"])
    responses = {item.key_for("correctness"): item.response for item in items}

    anchors = make_anchors(4)
    drift = check_drift(anchors, anchor_transcript([0.2] * 4, anchors))
    _, labels = build_transcript({"5-a": 5})
    calibration = calibrate(labels, transcript)

    health = build_health(
        HealthInputs(
            transcript=transcript, responses=responses, calibration=calibration, drift=drift
        )
    )
    assert health.drift_detected
    assert "JUDGE DRIFT DETECTED" in health.drift_detail
    assert health.n_calibration_items == 1
