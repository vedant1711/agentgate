"""Judge subsystem (L3): bias-controlled by construction, and audited rather than trusted."""

from agentgate.judge.audits import (
    POSITION_FLIP_THRESHOLD,
    VERBOSITY_THRESHOLD,
    Correlation,
    JudgeAudit,
    audit_scores,
    markdown_density,
    spearman,
)
from agentgate.judge.backed import TranscriptJudge
from agentgate.judge.backends import MalformedJudge, SyntheticJudge
from agentgate.judge.calibration import (
    KAPPA_GATE_FLOOR,
    CalibrationReport,
    CriterionAgreement,
    HumanLabel,
    LabelSet,
    calibrate,
    cohens_kappa,
    items_needing_labels,
    spearman_rho,
)
from agentgate.judge.drift import AnchorItem, AnchorSet, DriftReport, check_drift, record_band
from agentgate.judge.health import HealthInputs, audit_transcript, build_health
from agentgate.judge.independence import check_independence, model_family, same_family
from agentgate.judge.lockfile import DEFAULT_LOCK_PATH, JudgeLock
from agentgate.judge.rubric_judge import (
    DEFAULT_CRITERIA,
    JudgeConfig,
    RubricJudge,
    parse_pairwise_reply,
    parse_rubric_reply,
)
from agentgate.judge.rubrics import RUBRICS, Rubric, normalise, rubrics_hash
from agentgate.judge.transcript import (
    JudgedItem,
    JudgeEntry,
    JudgeSample,
    JudgeTranscript,
    PairwiseEntry,
    item_key,
)

__all__ = [
    "DEFAULT_CRITERIA",
    "DEFAULT_LOCK_PATH",
    "KAPPA_GATE_FLOOR",
    "POSITION_FLIP_THRESHOLD",
    "RUBRICS",
    "VERBOSITY_THRESHOLD",
    "AnchorItem",
    "AnchorSet",
    "CalibrationReport",
    "Correlation",
    "CriterionAgreement",
    "DriftReport",
    "HealthInputs",
    "HumanLabel",
    "JudgeAudit",
    "JudgeConfig",
    "JudgeEntry",
    "JudgeLock",
    "JudgeSample",
    "JudgeTranscript",
    "JudgedItem",
    "LabelSet",
    "MalformedJudge",
    "PairwiseEntry",
    "Rubric",
    "RubricJudge",
    "SyntheticJudge",
    "TranscriptJudge",
    "audit_scores",
    "audit_transcript",
    "build_health",
    "calibrate",
    "check_drift",
    "check_independence",
    "cohens_kappa",
    "item_key",
    "items_needing_labels",
    "markdown_density",
    "model_family",
    "normalise",
    "parse_pairwise_reply",
    "parse_rubric_reply",
    "record_band",
    "rubrics_hash",
    "same_family",
    "spearman",
    "spearman_rho",
]
