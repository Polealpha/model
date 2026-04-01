from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .schema import EpisodeWindowV1, episode_from_dict


SEQUENCE_FEATURES: Tuple[str, ...] = (
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


CONTEXT_VECTOR_KEYS: Tuple[str, ...] = (
    "big5_openness",
    "big5_conscientiousness",
    "big5_extraversion",
    "big5_agreeableness",
    "big5_neuroticism",
    "pref_directness",
    "pref_checkin_frequency",
    "pref_touchiness",
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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _rng_from_episode(episode_id: str) -> np.random.Generator:
    digest = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) % (2**32)
    return np.random.default_rng(seed)


def _to_episode(episode: EpisodeWindowV1 | Dict) -> EpisodeWindowV1:
    return episode if isinstance(episode, EpisodeWindowV1) else episode_from_dict(episode)


def episode_to_context_vector(episode: EpisodeWindowV1 | Dict) -> np.ndarray:
    ep = _to_episode(episode)
    values: List[float] = []
    for key in CONTEXT_VECTOR_KEYS:
        if key in ep.persona_profile:
            values.append(float(ep.persona_profile.get(key, 0.0) or 0.0))
        else:
            values.append(float(ep.context_flags.get(key, 0.0) or 0.0))
    return np.asarray(values, dtype=np.float32)


def episode_to_temporal_sequence(episode: EpisodeWindowV1 | Dict, steps: int = 12) -> np.ndarray:
    ep = _to_episode(episode)
    rng = _rng_from_episode(ep.episode_id)
    seq = np.zeros((steps, len(SEQUENCE_FEATURES)), dtype=np.float32)
    state = ep.state_labels
    obs = ep.observations
    context = ep.context_flags
    vulnerability = (
        0.30 * float(state.get("stress", 0.0))
        + 0.25 * float(state.get("fatigue", 0.0))
        + 0.20 * float(state.get("attention_drop", 0.0))
        + 0.15 * float(state.get("suppression", 0.0))
        + 0.10 * (1.0 - float(state.get("valence", 0.5)))
    )
    receptivity = float(obs.get("receptivity_proxy", 0.5))
    context_push = (
        0.18 * float(context.get("busy_speaking", 0.0))
        + 0.12 * float(context.get("privacy_on", 0.0))
        + 0.08 * float(context.get("quiet_mode", 0.0))
    )

    center = (steps - 1) / 2.0
    for t in range(steps):
        pos = (t - center) / max(1.0, center)
        phase = 2.0 * math.pi * (t / max(1, steps - 1))
        for j, key in enumerate(SEQUENCE_FEATURES):
            base = float(obs.get(key, 0.0))
            wave = math.sin(phase + 0.35 * j) + 0.5 * math.cos(2.0 * phase + 0.17 * j)
            amp = 0.03 + 0.18 * vulnerability + 0.05 * abs(receptivity - 0.5)
            trend = 0.10 * vulnerability - 0.08 * context_push
            if key in {"stress_proxy", "prosody_stress", "fidget_score"}:
                trend += 0.06 * vulnerability
            if key in {"silence_ratio", "fatigue_proxy", "posture_slouch_score"}:
                trend += 0.04 * float(state.get("fatigue", 0.0))
            if key in {"receptivity_proxy", "voice_energy"}:
                trend -= 0.05 * context_push
            noise = rng.normal(0.0, 0.015)
            value = _clamp01(base + amp * wave * 0.5 + trend * pos + noise)
            seq[t, j] = value
    return seq


def build_sequence_batch(
    episodes: Sequence[EpisodeWindowV1 | Dict],
    steps: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    sequences = [episode_to_temporal_sequence(ep, steps=steps) for ep in episodes]
    context = [episode_to_context_vector(ep) for ep in episodes]
    return np.stack(sequences).astype(np.float32), np.stack(context).astype(np.float32)

