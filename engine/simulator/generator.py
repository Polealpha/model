from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List

from engine.policy.rule_baseline import decide_strategy, decide_timing
from engine.schemas.contracts import (
    ContextFlagsV1,
    EpisodeWindowV1,
    FeedbackEventV1,
    PersonaProfileV1,
)
from engine.state.estimator import estimate_state_from_features


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def sample_persona(index: int, rng: random.Random) -> PersonaProfileV1:
    big_five = [_clamp01(rng.uniform(0.15, 0.85)) for _ in range(5)]
    return PersonaProfileV1(
        persona_id=f"persona_{index:04d}",
        big_five=big_five,
        interaction_preferences={
            "interruption_tolerance": _clamp01(0.4 + 0.35 * big_five[0] - 0.2 * big_five[4] + rng.uniform(-0.1, 0.1)),
            "care_seeking": _clamp01(0.3 + 0.30 * big_five[3] + 0.20 * big_five[4] + rng.uniform(-0.1, 0.1)),
            "privacy_sensitivity": _clamp01(0.35 + 0.35 * big_five[4] + rng.uniform(-0.1, 0.1)),
        },
        notes={"source": "synthetic_big_five"},
    )


def sample_context(rng: random.Random) -> ContextFlagsV1:
    return ContextFlagsV1(
        busy_speaking=rng.random() < 0.22,
        privacy_on=rng.random() < 0.05,
        quiet_mode=rng.random() < 0.08,
        daily_count=rng.randint(0, 4),
        cooldown_active=rng.random() < 0.12,
        scene="desk",
    )


def sample_feature_vector(rng: random.Random, trajectory_phase: float) -> Dict[str, float]:
    wave = 0.5 + 0.5 * math.sin(trajectory_phase)
    return {
        "posture_slouch": _clamp01(0.2 + 0.6 * wave + rng.uniform(-0.15, 0.15)),
        "stillness": _clamp01(0.1 + 0.7 * wave + rng.uniform(-0.10, 0.10)),
        "motion_energy": _clamp01(0.8 - 0.5 * wave + rng.uniform(-0.15, 0.15)),
        "voice_tension": _clamp01(0.2 + 0.6 * wave + rng.uniform(-0.15, 0.15)),
        "fidget": _clamp01(0.2 + 0.5 * (1.0 - wave) + rng.uniform(-0.12, 0.12)),
        "gaze_variance": _clamp01(0.2 + 0.5 * wave + rng.uniform(-0.12, 0.12)),
        "event_pressure": _clamp01(0.15 + 0.65 * wave + rng.uniform(-0.12, 0.12)),
        "attention_frag": _clamp01(0.1 + 0.5 * wave + rng.uniform(-0.12, 0.12)),
        "blink_irregularity": _clamp01(0.15 + 0.45 * wave + rng.uniform(-0.12, 0.12)),
        "expression_flatness": _clamp01(0.15 + 0.55 * wave + rng.uniform(-0.12, 0.12)),
        "voice_mismatch": _clamp01(0.10 + 0.50 * wave + rng.uniform(-0.10, 0.10)),
        "social_masking": _clamp01(0.10 + 0.60 * wave + rng.uniform(-0.10, 0.10)),
        "voice_energy": _clamp01(0.20 + 0.45 * (1.0 - wave) + rng.uniform(-0.10, 0.10)),
        "positive_affect": _clamp01(0.70 - 0.45 * wave + rng.uniform(-0.10, 0.10)),
    }


def synthesize_episode(sample_id: str, persona: PersonaProfileV1, rng: random.Random) -> EpisodeWindowV1:
    context = sample_context(rng)
    phase = rng.uniform(0.0, 2.0 * math.pi)
    features = sample_feature_vector(rng, phase)
    state = estimate_state_from_features(features, context.busy_speaking)
    window = EpisodeWindowV1(
        sample_id=sample_id,
        video_clip_path=f"synthetic://video/{sample_id}.mp4",
        audio_clip_path=f"synthetic://audio/{sample_id}.wav",
        timestamp_range=(0, 60000),
        persona_profile=persona,
        context_flags=context,
        feature_vector=features,
        state_labels=state,
        timing_label="none",
        strategy_label="observe",
        script_template_id="default_observe",
        feedback={},
        source="synthetic",
    )
    timing = decide_timing(window)
    strategy = decide_strategy(window, timing)
    window.timing_label = timing.decision
    window.strategy_label = strategy.strategy_level
    window.script_template_id = strategy.strategy_level
    feedback = FeedbackEventV1(
        accepted=timing.decision == "immediate" and rng.random() < 0.7,
        ignored=timing.decision != "immediate" and rng.random() < 0.5,
        annoyed=context.busy_speaking and timing.decision == "immediate" and rng.random() < 0.6,
        response_latency_ms=rng.randint(500, 12000),
        self_report_optional={"synthetic": True},
    )
    window.feedback = feedback.to_dict()
    return window


def generate_dataset(num_samples: int, seed: int = 7) -> Iterable[EpisodeWindowV1]:
    rng = random.Random(seed)
    personas: List[PersonaProfileV1] = [sample_persona(i, rng) for i in range(max(32, num_samples // 8))]
    for index in range(num_samples):
        persona = personas[index % len(personas)]
        yield synthesize_episode(f"episode_{index:06d}", persona, rng)


def write_jsonl(samples: Iterable[EpisodeWindowV1], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

