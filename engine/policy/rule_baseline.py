from __future__ import annotations

from typing import Dict, Tuple

from engine.schemas.contracts import (
    CareUtteranceV1,
    EpisodeWindowV1,
    StrategyLabel,
    StrategyPlanV1,
    TimingDecisionV1,
    TimingLabel,
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _persona_preference(persona: Dict[str, float], key: str, default: float) -> float:
    return float(persona.get(key, default))


def compute_receptivity(window: EpisodeWindowV1) -> Tuple[float, Dict[str, float]]:
    state = window.state_labels
    prefs = window.persona_profile.interaction_preferences
    interrupt_tolerance = _persona_preference(prefs, "interruption_tolerance", 0.5)
    care_seeking = _persona_preference(prefs, "care_seeking", 0.5)
    privacy_sensitivity = _persona_preference(prefs, "privacy_sensitivity", 0.5)
    features = window.feature_vector

    vulnerability = _clamp01(
        0.28 * state.stress
        + 0.24 * state.fatigue
        + 0.22 * state.attention_drop
        + 0.18 * state.suppression
        + 0.08 * (1.0 - state.valence)
    )
    interrupt_cost = _clamp01(
        0.35 * float(window.context_flags.busy_speaking)
        + 0.20 * float(window.context_flags.cooldown_active)
        + 0.15 * min(1.0, window.context_flags.daily_count / 5.0)
        + 0.15 * features.get("event_pressure", 0.0)
        + 0.15 * privacy_sensitivity
    )
    receptivity = _clamp01(
        0.55 * vulnerability
        + 0.20 * care_seeking
        + 0.15 * interrupt_tolerance
        - 0.35 * interrupt_cost
    )
    return receptivity, {
        "vulnerability": vulnerability,
        "interrupt_cost": interrupt_cost,
        "care_seeking": care_seeking,
        "interruption_tolerance": interrupt_tolerance,
        "privacy_sensitivity": privacy_sensitivity,
    }


def decide_timing(window: EpisodeWindowV1) -> TimingDecisionV1:
    reasons = []
    if window.context_flags.privacy_on:
        return TimingDecisionV1(
            decision=TimingLabel.NONE.value,
            score=0.0,
            uncertainty=0.05,
            why_codes=["privacy_on"],
        )
    if window.context_flags.quiet_mode or window.context_flags.cooldown_active:
        return TimingDecisionV1(
            decision=TimingLabel.DELAY.value,
            score=0.15,
            uncertainty=0.10,
            why_codes=["quiet_or_cooldown"],
        )

    receptivity, parts = compute_receptivity(window)
    vulnerability = parts["vulnerability"]
    interrupt_cost = parts["interrupt_cost"]
    uncertainty = _clamp01(0.55 - abs(receptivity - 0.5))

    if interrupt_cost > 0.75:
        reasons.append("high_interrupt_cost")
        return TimingDecisionV1(
            decision=TimingLabel.NONE.value,
            score=receptivity,
            uncertainty=uncertainty,
            why_codes=reasons,
        )
    if vulnerability > 0.7 and receptivity > 0.58:
        reasons.extend(["high_vulnerability", "receptive_now"])
        return TimingDecisionV1(
            decision=TimingLabel.IMMEDIATE.value,
            score=receptivity,
            uncertainty=uncertainty,
            why_codes=reasons,
        )
    if vulnerability > 0.45:
        reasons.extend(["moderate_vulnerability", "wait_and_check"])
        return TimingDecisionV1(
            decision=TimingLabel.DELAY.value,
            score=receptivity,
            uncertainty=uncertainty,
            why_codes=reasons,
        )
    reasons.append("below_threshold")
    return TimingDecisionV1(
        decision=TimingLabel.NONE.value,
        score=receptivity,
        uncertainty=uncertainty,
        why_codes=reasons,
    )


def decide_strategy(window: EpisodeWindowV1, timing: TimingDecisionV1) -> StrategyPlanV1:
    state = window.state_labels
    if timing.decision == TimingLabel.NONE.value:
        return StrategyPlanV1(
            strategy_level=StrategyLabel.OBSERVE.value,
            outline_steps=["observe", "log", "wait"],
            utterance_constraints={"speak": False},
        )
    if timing.decision == TimingLabel.DELAY.value:
        return StrategyPlanV1(
            strategy_level=StrategyLabel.NUDGE.value,
            outline_steps=["observe", "light_check_later", "confirm_if_persistent"],
            utterance_constraints={"tone": "light", "max_sentences": 1},
        )
    if state.stress > 0.82 or state.suppression > 0.72:
        level = StrategyLabel.GUARD.value
        steps = ["describe_observation", "gentle_hypothesis", "confirm", "repair_or_support"]
    else:
        level = StrategyLabel.CARE.value
        steps = ["describe_observation", "gentle_hypothesis", "confirm"]
    return StrategyPlanV1(
        strategy_level=level,
        outline_steps=steps,
        utterance_constraints={"tone": "warm", "max_sentences": 2},
    )


def render_utterance(window: EpisodeWindowV1, strategy: StrategyPlanV1) -> CareUtteranceV1:
    if strategy.strategy_level == StrategyLabel.OBSERVE.value:
        return CareUtteranceV1(draft_text="", confirmation_question="", repair_text="")

    observation = "我注意到你刚刚状态有些不太一样"
    if window.state_labels.fatigue >= max(window.state_labels.stress, window.state_labels.suppression):
        observation = "我注意到你刚刚好像有点累，动作也慢下来了"
    elif window.state_labels.suppression > 0.65:
        observation = "我注意到你看起来很平静，但好像一直在压着什么"
    elif window.state_labels.stress > 0.65:
        observation = "我注意到你刚刚有点绷着，像是在扛着压力"

    hypothesis = "也可能是我看错了"
    confirm = "如果你愿意，我可以先陪你确认一下现在更需要安静，还是更想被接住一点。"
    repair = "如果我打扰到你了，你直接告诉我就行，我先退开。"
    draft = f"{observation}，{hypothesis}。{confirm}"
    return CareUtteranceV1(
        draft_text=draft,
        confirmation_question=confirm,
        repair_text=repair,
    )

