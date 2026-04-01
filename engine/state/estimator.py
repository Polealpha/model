from __future__ import annotations

from typing import Dict

from engine.schemas.contracts import EpisodeWindowV1, StateVectorV1


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def estimate_state_from_features(features: Dict[str, float], busy_speaking: bool) -> StateVectorV1:
    fatigue = _clamp01(
        0.40 * features.get("posture_slouch", 0.0)
        + 0.35 * features.get("stillness", 0.0)
        + 0.25 * (1.0 - features.get("motion_energy", 0.5))
    )
    stress = _clamp01(
        0.32 * features.get("voice_tension", 0.0)
        + 0.26 * features.get("fidget", 0.0)
        + 0.22 * features.get("gaze_variance", 0.0)
        + 0.20 * features.get("event_pressure", 0.0)
    )
    attention_drop = _clamp01(
        0.45 * features.get("attention_frag", 0.0)
        + 0.30 * features.get("stillness", 0.0)
        + 0.25 * features.get("blink_irregularity", 0.0)
    )
    suppression = _clamp01(
        0.45 * features.get("expression_flatness", 0.0)
        + 0.30 * features.get("voice_mismatch", 0.0)
        + 0.25 * features.get("social_masking", 0.0)
    )
    arousal = _clamp01(
        0.40 * stress
        + 0.25 * features.get("motion_energy", 0.0)
        + 0.20 * features.get("voice_energy", 0.0)
        + 0.15 * float(busy_speaking)
    )
    valence = _clamp01(
        0.55
        - 0.30 * stress
        - 0.25 * fatigue
        - 0.20 * suppression
        + 0.20 * features.get("positive_affect", 0.0)
    )
    return StateVectorV1(
        valence=valence,
        arousal=arousal,
        stress=stress,
        fatigue=fatigue,
        attention_drop=attention_drop,
        suppression=suppression,
    )


def attach_estimated_state(window: EpisodeWindowV1) -> EpisodeWindowV1:
    window.state_labels = estimate_state_from_features(
        window.feature_vector,
        busy_speaking=window.context_flags.busy_speaking,
    )
    return window

