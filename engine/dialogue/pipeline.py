from __future__ import annotations

from engine.policy.rule_baseline import decide_strategy, decide_timing, render_utterance
from engine.schemas.contracts import EpisodeWindowV1


def run_dialogue_pipeline(window: EpisodeWindowV1) -> dict:
    timing = decide_timing(window)
    strategy = decide_strategy(window, timing)
    utterance = render_utterance(window, strategy)
    return {
        "timing_decision": timing.to_dict(),
        "strategy_plan": strategy.to_dict(),
        "care_utterance": utterance.to_dict(),
    }

