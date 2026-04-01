from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


TIMING_LABELS = ("none", "delay", "immediate")
STRATEGY_LABELS = ("observe", "nudge", "care", "guard")
STATE_LABELS = ("valence", "arousal", "stress", "fatigue", "attention_drop", "suppression")


@dataclass
class StateVectorV1:
    valence: float
    arousal: float
    stress: float
    fatigue: float
    attention_drop: float
    suppression: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class TimingDecisionV1:
    decision: str
    score: float
    uncertainty: float
    why_codes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyPlanV1:
    strategy_level: str
    outline_steps: List[str]
    utterance_constraints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CareUtteranceV1:
    draft_text: str
    confirmation_question: str
    repair_text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FeedbackEventV1:
    accepted: bool
    ignored: bool
    annoyed: bool
    response_latency_ms: Optional[int] = None
    self_report_optional: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeWindowV1:
    episode_id: str
    video_clip_path: str
    audio_clip_path: str
    timestamp_range: Tuple[int, int]
    persona_profile: Dict[str, Any]
    context_flags: Dict[str, Any]
    state_labels: Dict[str, float]
    timing_label: str
    strategy_label: str
    script_template_id: str
    feedback: Dict[str, Any] = field(default_factory=dict)
    observations: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["timestamp_range"] = list(self.timestamp_range)
        return payload


def episode_from_dict(payload: Dict[str, Any]) -> EpisodeWindowV1:
    return EpisodeWindowV1(
        episode_id=str(payload.get("episode_id", "")),
        video_clip_path=str(payload.get("video_clip_path", "")),
        audio_clip_path=str(payload.get("audio_clip_path", "")),
        timestamp_range=tuple(payload.get("timestamp_range", (0, 0)))[:2],
        persona_profile=dict(payload.get("persona_profile", {})),
        context_flags=dict(payload.get("context_flags", {})),
        state_labels=dict(payload.get("state_labels", {})),
        timing_label=str(payload.get("timing_label", "none")),
        strategy_label=str(payload.get("strategy_label", "observe")),
        script_template_id=str(payload.get("script_template_id", "default")),
        feedback=dict(payload.get("feedback", {})),
        observations=dict(payload.get("observations", {})),
        extra=dict(payload.get("extra", {})),
    )

