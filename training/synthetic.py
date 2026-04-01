from __future__ import annotations

from dataclasses import dataclass
from math import exp
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .data import save_episodes
from .io import save_json
from .schema import EpisodeWindowV1, STATE_LABELS


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


@dataclass
class SyntheticGeneratorConfig:
    seed: int = 42
    window_sec: int = 60
    step_sec: int = 5


class SyntheticDatasetGenerator:
    def __init__(self, cfg: SyntheticGeneratorConfig | Dict | None = None) -> None:
        if cfg is None:
            cfg = SyntheticGeneratorConfig()
        elif isinstance(cfg, dict):
            cfg = SyntheticGeneratorConfig(
                seed=int(cfg.get("seed", 42)),
                window_sec=int(cfg.get("window_sec", 60)),
                step_sec=int(cfg.get("step_sec", 5)),
            )
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def generate(self, total: int, split_name: str = "train") -> List[EpisodeWindowV1]:
        episodes: List[EpisodeWindowV1] = []
        for idx in range(total):
            episodes.append(self._sample_episode(idx=idx, split_name=split_name))
        return episodes

    def _sample_episode(self, idx: int, split_name: str) -> EpisodeWindowV1:
        persona = self._sample_persona()
        context = self._sample_context(persona)
        state = self._sample_state(persona, context)
        observations = self._sample_observations(persona, context, state)
        timing_label, timing_score = self._label_timing(persona, context, state, observations)
        strategy_label = self._label_strategy(persona, context, state, timing_label)
        feedback = self._sample_feedback(persona, context, state, timing_label, observations)
        template_id = self._template_for(strategy_label, timing_label)
        episode_id = f"{split_name}_{idx:06d}"
        start_ts = idx * self.cfg.step_sec * 1000
        end_ts = start_ts + self.cfg.window_sec * 1000
        return EpisodeWindowV1(
            episode_id=episode_id,
            video_clip_path=f"synthetic://{episode_id}/video.mp4",
            audio_clip_path=f"synthetic://{episode_id}/audio.wav",
            timestamp_range=(start_ts, end_ts),
            persona_profile=persona,
            context_flags=context,
            state_labels=state,
            timing_label=timing_label,
            strategy_label=strategy_label,
            script_template_id=template_id,
            feedback=feedback,
            observations=observations,
            extra={"timing_score": timing_score, "split": split_name},
        )

    def _sample_persona(self) -> Dict[str, float]:
        base = self.rng.beta(2.0, 2.0, size=5)
        return {
            "big5_openness": float(base[0]),
            "big5_conscientiousness": float(base[1]),
            "big5_extraversion": float(base[2]),
            "big5_agreeableness": float(base[3]),
            "big5_neuroticism": float(base[4]),
            "pref_directness": float(clamp01(0.25 + 0.6 * self.rng.random())),
            "pref_checkin_frequency": float(clamp01(0.20 + 0.7 * self.rng.random())),
            "pref_touchiness": float(clamp01(0.10 + 0.5 * self.rng.random())),
        }

    def _sample_context(self, persona: Dict[str, float]) -> Dict[str, float]:
        busy = float(self.rng.random() < 0.42)
        privacy = float(self.rng.random() < 0.12)
        quiet = float(self.rng.random() < 0.18)
        daily_count = float(self.rng.integers(0, 6))
        cooldown_active = float(self.rng.random() < 0.08)
        meeting_pressure = clamp01(0.2 + 0.5 * busy + 0.2 * self.rng.random())
        social_load = clamp01(0.15 + 0.4 * persona["big5_extraversion"] + 0.3 * self.rng.random())
        time_of_day = float(self.rng.uniform(0, 1))
        recent_rejections = float(self.rng.integers(0, 4))
        recent_accepts = float(self.rng.integers(0, 4))
        return {
            "busy_speaking": busy,
            "privacy_on": privacy,
            "quiet_mode": quiet,
            "daily_count": daily_count,
            "cooldown_active": cooldown_active,
            "meeting_pressure": meeting_pressure,
            "social_load": social_load,
            "time_of_day": time_of_day,
            "recent_rejections": recent_rejections,
            "recent_accepts": recent_accepts,
        }

    def _sample_state(self, persona: Dict[str, float], context: Dict[str, float]) -> Dict[str, float]:
        busy = context["busy_speaking"]
        privacy = context["privacy_on"]
        quiet = context["quiet_mode"]
        neuroticism = persona["big5_neuroticism"]
        conscientiousness = persona["big5_conscientiousness"]
        extraversion = persona["big5_extraversion"]

        stress = clamp01(0.12 + 0.44 * neuroticism + 0.20 * busy + 0.10 * context["meeting_pressure"] + 0.08 * self.rng.normal())
        fatigue = clamp01(0.15 + 0.35 * (1.0 - conscientiousness) + 0.18 * quiet + 0.16 * context["daily_count"] / 5.0 + 0.08 * self.rng.normal())
        attention_drop = clamp01(0.10 + 0.55 * fatigue + 0.15 * privacy + 0.12 * self.rng.normal())
        suppression = clamp01(0.08 + 0.45 * privacy + 0.15 * (1.0 - extraversion) + 0.08 * self.rng.normal())
        valence = clamp01(0.78 - 0.45 * stress - 0.20 * fatigue + 0.10 * persona["big5_agreeableness"] + 0.06 * self.rng.normal())
        arousal = clamp01(0.35 + 0.35 * stress + 0.20 * attention_drop + 0.10 * self.rng.normal())
        return {
            "valence": valence,
            "arousal": arousal,
            "stress": stress,
            "fatigue": fatigue,
            "attention_drop": attention_drop,
            "suppression": suppression,
        }

    def _sample_observations(self, persona: Dict[str, float], context: Dict[str, float], state: Dict[str, float]) -> Dict[str, float]:
        stress = state["stress"]
        fatigue = state["fatigue"]
        attention_drop = state["attention_drop"]
        suppression = state["suppression"]
        busy = context["busy_speaking"]
        privacy = context["privacy_on"]
        quiet = context["quiet_mode"]

        receptivity_proxy = clamp01(0.65 - 0.40 * busy - 0.32 * privacy - 0.18 * quiet + 0.25 * persona["big5_agreeableness"] + 0.15 * persona["pref_checkin_frequency"] + 0.08 * self.rng.normal())
        face_presence_ratio = clamp01(0.92 - 0.48 * attention_drop - 0.10 * quiet + 0.07 * self.rng.normal())
        gaze_avert_ratio = clamp01(0.18 + 0.56 * attention_drop + 0.08 * privacy + 0.06 * self.rng.normal())
        head_motion_var = clamp01(0.16 + 0.42 * fatigue + 0.16 * stress + 0.08 * self.rng.normal())
        posture_slouch_score = clamp01(0.18 + 0.52 * fatigue + 0.12 * busy + 0.07 * self.rng.normal())
        fidget_score = clamp01(0.10 + 0.48 * stress + 0.24 * persona["big5_extraversion"] + 0.08 * self.rng.normal())
        voice_energy = clamp01(0.86 - 0.48 * fatigue - 0.20 * stress + 0.08 * self.rng.normal())
        speech_rate = clamp01(0.55 + 0.22 * persona["big5_extraversion"] - 0.18 * fatigue + 0.06 * self.rng.normal())
        silence_ratio = clamp01(0.14 + 0.34 * fatigue + 0.26 * privacy + 0.10 * self.rng.normal())
        prosody_stress = clamp01(0.12 + 0.56 * stress + 0.12 * suppression + 0.08 * self.rng.normal())
        attention_drop_proxy = clamp01(attention_drop + 0.06 * self.rng.normal())
        fatigue_proxy = clamp01(fatigue + 0.06 * self.rng.normal())
        stress_proxy = clamp01(stress + 0.06 * self.rng.normal())

        return {
            "face_presence_ratio": face_presence_ratio,
            "gaze_avert_ratio": gaze_avert_ratio,
            "head_motion_var": head_motion_var,
            "posture_slouch_score": posture_slouch_score,
            "fidget_score": fidget_score,
            "voice_energy": voice_energy,
            "speech_rate": speech_rate,
            "silence_ratio": silence_ratio,
            "prosody_stress": prosody_stress,
            "attention_drop_proxy": attention_drop_proxy,
            "fatigue_proxy": fatigue_proxy,
            "stress_proxy": stress_proxy,
            "receptivity_proxy": receptivity_proxy,
        }

    def _label_timing(
        self,
        persona: Dict[str, float],
        context: Dict[str, float],
        state: Dict[str, float],
        obs: Dict[str, float],
    ) -> tuple[str, float]:
        busy = context["busy_speaking"]
        privacy = context["privacy_on"]
        quiet = context["quiet_mode"]
        receptivity = obs["receptivity_proxy"]
        vulnerability = 0.45 * state["stress"] + 0.25 * state["fatigue"] + 0.20 * state["attention_drop"] + 0.10 * state["suppression"]
        score = clamp01(
            0.58 * vulnerability
            + 0.30 * receptivity
            + 0.12 * persona["pref_checkin_frequency"]
            - 0.18 * busy
            - 0.15 * privacy
            - 0.05 * quiet
            + 0.05 * self.rng.normal()
        )
        if privacy > 0.5 and vulnerability < 0.76:
            return "none", score
        if busy > 0.5 and vulnerability < 0.60 and receptivity < 0.52:
            return "none", score
        if (vulnerability >= 0.58 and receptivity >= 0.36 and privacy < 0.5) or (score >= 0.54 and vulnerability >= 0.44):
            return "immediate", score
        if score >= 0.43:
            return "delay", score
        return "none", score

    def _label_strategy(self, persona: Dict[str, float], context: Dict[str, float], state: Dict[str, float], timing_label: str) -> str:
        if context["privacy_on"] > 0.5 or (context["cooldown_active"] > 0.5 and timing_label != "immediate"):
            return "guard"
        if timing_label == "none":
            return "observe"
        if timing_label == "delay":
            return "nudge"
        if (
            state["stress"] >= 0.58
            or state["suppression"] >= 0.48
            or state["fatigue"] >= 0.72
            or state["attention_drop"] >= 0.74
        ):
            return "care"
        return "nudge"

    def _sample_feedback(
        self,
        persona: Dict[str, float],
        context: Dict[str, float],
        state: Dict[str, float],
        timing_label: str,
        obs: Dict[str, float],
    ) -> Dict[str, float]:
        true_need = clamp01(0.30 + 0.40 * state["stress"] + 0.25 * state["fatigue"] + 0.12 * state["attention_drop"] + 0.10 * state["suppression"])
        align = 1.0 if timing_label == ("immediate" if true_need > 0.67 else "delay" if true_need > 0.45 else "none") else 0.0
        acceptance = clamp01(0.25 + 0.45 * obs["receptivity_proxy"] + 0.20 * persona["pref_checkin_frequency"] + 0.20 * align - 0.10 * context["busy_speaking"])
        annoyance = clamp01(0.15 + 0.35 * context["busy_speaking"] + 0.20 * context["privacy_on"] + 0.25 * (1.0 - align) - 0.12 * obs["receptivity_proxy"])
        helpfulness = clamp01(0.20 + 0.45 * true_need + 0.20 * align + 0.12 * persona["big5_agreeableness"] - 0.14 * annoyance)
        accepted = acceptance >= 0.55
        ignored = acceptance < 0.45 and annoyance < 0.55
        annoyed = annoyance >= 0.55
        return {
            "accepted": float(accepted),
            "ignored": float(ignored),
            "annoyed": float(annoyed),
            "response_latency_ms": float(800 + 2200 * (1.0 - acceptance) + 800 * annoyance),
            "helpfulness": helpfulness,
            "acceptance": acceptance,
            "annoyance": annoyance,
        }

    def _template_for(self, strategy_label: str, timing_label: str) -> str:
        if strategy_label == "guard":
            return "privacy_guard"
        if timing_label == "none":
            return "observe_only"
        if strategy_label == "nudge":
            return "gentle_checkin"
        if strategy_label == "care":
            return "warm_care"
        return "observe_only"


def write_synthetic_dataset(root: str | Path, cfg: Dict[str, float | int], counts: Dict[str, int]) -> Dict[str, str]:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    generator = SyntheticDatasetGenerator(cfg)
    outputs: Dict[str, str] = {}
    manifest = {
        "config": cfg,
        "counts": counts,
        "label_space": {
            "timing": ["none", "delay", "immediate"],
            "strategy": ["observe", "nudge", "care", "guard"],
            "state": list(STATE_LABELS),
        },
    }
    for split, count in counts.items():
        episodes = generator.generate(int(count), split_name=split)
        file_path = root_path / f"{split}.jsonl"
        save_episodes(file_path, episodes)
        outputs[split] = str(file_path)
    save_json(root_path / "manifest.json", manifest)
    return outputs
