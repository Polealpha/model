from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .schema import EpisodeWindowV1, STATE_LABELS, STRATEGY_LABELS, TIMING_LABELS, episode_from_dict


DEFAULT_PERSONA_FEATURES = (
    "big5_openness",
    "big5_conscientiousness",
    "big5_extraversion",
    "big5_agreeableness",
    "big5_neuroticism",
    "pref_directness",
    "pref_checkin_frequency",
    "pref_touchiness",
)

DEFAULT_CONTEXT_FEATURES = (
    "busy_speaking",
    "privacy_on",
    "quiet_mode",
    "daily_count",
    "cooldown_active",
    "meeting_pressure",
    "social_load",
    "time_of_day",
    "recent_rejections",
    "recent_accepts",
)

DEFAULT_OBSERVATION_FEATURES = (
    "face_presence_ratio",
    "gaze_avert_ratio",
    "head_motion_var",
    "posture_slouch_score",
    "fidget_score",
    "voice_energy",
    "speech_rate",
    "silence_ratio",
    "prosody_stress",
    "attention_drop_proxy",
    "fatigue_proxy",
    "stress_proxy",
    "receptivity_proxy",
)


def episode_to_feature_dict(episode: EpisodeWindowV1 | Dict) -> Dict[str, float]:
    if not isinstance(episode, EpisodeWindowV1):
        episode = episode_from_dict(episode)

    features: Dict[str, float] = {}
    persona = episode.persona_profile
    context = episode.context_flags
    obs = episode.observations

    for key in DEFAULT_PERSONA_FEATURES:
        value = persona.get(key, persona.get(key.replace("big5_", ""), 0.0))
        features[key] = float(value if value is not None else 0.0)
    for key in DEFAULT_CONTEXT_FEATURES:
        value = context.get(key, 0.0)
        features[key] = float(value if value is not None else 0.0)
    for key in DEFAULT_OBSERVATION_FEATURES:
        value = obs.get(key, 0.0)
        features[key] = float(value if value is not None else 0.0)

    if episode.timestamp_range:
        start, end = episode.timestamp_range
        features["window_ms"] = float(max(0, end - start))
    else:
        features["window_ms"] = 0.0

    return features


def build_feature_matrix(episodes: Sequence[EpisodeWindowV1 | Dict]) -> Tuple[np.ndarray, List[str]]:
    dicts = [episode_to_feature_dict(item) for item in episodes]
    feature_names = sorted({name for row in dicts for name in row})
    X = np.array([[row.get(name, 0.0) for name in feature_names] for row in dicts], dtype=np.float32)
    return X, feature_names


def extract_targets(episodes: Sequence[EpisodeWindowV1 | Dict]) -> Dict[str, np.ndarray]:
    episodes = [episode_from_dict(ep) if not isinstance(ep, EpisodeWindowV1) else ep for ep in episodes]
    timing = np.array([TIMING_LABELS.index(ep.timing_label) if ep.timing_label in TIMING_LABELS else 0 for ep in episodes], dtype=np.int64)
    strategy = np.array([STRATEGY_LABELS.index(ep.strategy_label) if ep.strategy_label in STRATEGY_LABELS else 0 for ep in episodes], dtype=np.int64)
    state = np.array(
        [[float(ep.state_labels.get(name, 0.0)) for name in STATE_LABELS] for ep in episodes],
        dtype=np.float32,
    )
    template_ids = np.array([str(ep.script_template_id) for ep in episodes], dtype=object)
    return {"timing": timing, "strategy": strategy, "state": state, "template": template_ids}


def feature_summary(X: np.ndarray) -> Dict[str, float]:
    if X.size == 0:
        return {"rows": 0.0, "cols": 0.0}
    return {
        "rows": float(X.shape[0]),
        "cols": float(X.shape[1]),
        "mean": float(np.mean(X)),
        "std": float(np.std(X)),
    }

