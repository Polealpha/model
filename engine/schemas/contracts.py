from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class TimingLabel(str, Enum):
    IMMEDIATE = "immediate"
    DELAY = "delay"
    NONE = "none"


class StrategyLabel(str, Enum):
    OBSERVE = "observe"
    NUDGE = "nudge"
    CARE = "care"
    GUARD = "guard"


@dataclass
class PersonaProfileV1:
    persona_id: str
    big_five: List[float]
    interaction_preferences: Dict[str, float]
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContextFlagsV1:
    busy_speaking: bool
    privacy_on: bool
    quiet_mode: bool
    daily_count: int
    cooldown_active: bool
    scene: str = "desk"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StateVectorV1:
    valence: float
    arousal: float
    stress: float
    fatigue: float
    attention_drop: float
    suppression: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeWindowV1:
    sample_id: str
    video_clip_path: str
    audio_clip_path: str
    timestamp_range: Tuple[int, int]
    persona_profile: PersonaProfileV1
    context_flags: ContextFlagsV1
    feature_vector: Dict[str, float]
    state_labels: StateVectorV1
    timing_label: str
    strategy_label: str
    script_template_id: str
    feedback: Dict[str, Any] = field(default_factory=dict)
    source: str = "synthetic"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["persona_profile"] = self.persona_profile.to_dict()
        payload["context_flags"] = self.context_flags.to_dict()
        payload["state_labels"] = self.state_labels.to_dict()
        return payload


@dataclass
class TimingDecisionV1:
    decision: str
    score: float
    uncertainty: float
    why_codes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyPlanV1:
    strategy_level: str
    outline_steps: List[str]
    utterance_constraints: Dict[str, Any]

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
    response_latency_ms: int
    self_report_optional: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

