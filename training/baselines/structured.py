from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..feature_utils import build_feature_matrix, extract_targets
from ..schema import CareUtteranceV1, EpisodeWindowV1, StrategyPlanV1, TimingDecisionV1, STRATEGY_LABELS, TIMING_LABELS


@dataclass
class StructuredBaselineConfig:
    C: float = 2.0
    max_iter: int = 1000
    seed: int = 42


class StructuredBaseline:
    def __init__(self, config: StructuredBaselineConfig | Dict[str, Any] | None = None) -> None:
        if config is None:
            config = StructuredBaselineConfig()
        elif isinstance(config, dict):
            config = StructuredBaselineConfig(**config)
        self.config = config
        self.timing_model: Optional[Pipeline] = None
        self.strategy_model: Optional[Pipeline] = None
        self.feature_names: List[str] = []

    def fit(self, episodes: Sequence[EpisodeWindowV1 | Dict[str, Any]]) -> "StructuredBaseline":
        X, feature_names = build_feature_matrix(episodes)
        targets = extract_targets(episodes)
        self.feature_names = feature_names
        self.timing_model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.config.C,
                        max_iter=self.config.max_iter,
                        class_weight="balanced",
                        random_state=self.config.seed,
                    ),
                ),
            ]
        )
        self.strategy_model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.config.C,
                        max_iter=self.config.max_iter,
                        class_weight="balanced",
                        random_state=self.config.seed,
                    ),
                ),
            ]
        )
        self.timing_model.fit(X, targets["timing"])
        self.strategy_model.fit(X, targets["strategy"])
        return self

    def predict_proba(self, episodes: Sequence[EpisodeWindowV1 | Dict[str, Any]]) -> Dict[str, np.ndarray]:
        if self.timing_model is None or self.strategy_model is None:
            raise RuntimeError("StructuredBaseline must be fit before prediction")
        X, _ = build_feature_matrix(episodes)
        return {
            "timing": self.timing_model.predict_proba(X),
            "strategy": self.strategy_model.predict_proba(X),
        }

    def predict(self, episodes: Sequence[EpisodeWindowV1 | Dict[str, Any]]) -> List[Dict[str, Any]]:
        probs = self.predict_proba(episodes)
        timing_idx = np.argmax(probs["timing"], axis=1)
        strategy_idx = np.argmax(probs["strategy"], axis=1)
        output: List[Dict[str, Any]] = []
        for i in range(len(timing_idx)):
            timing_label = TIMING_LABELS[int(timing_idx[i])]
            strategy_label = STRATEGY_LABELS[int(strategy_idx[i])]
            timing_score = float(np.max(probs["timing"][i]))
            strategy_score = float(np.max(probs["strategy"][i]))
            output.append(
                {
                    "timing": TimingDecisionV1(
                        decision=timing_label,
                        score=timing_score,
                        uncertainty=float(1.0 - timing_score),
                        why_codes=["structured_baseline"],
                    ).to_dict(),
                    "strategy": StrategyPlanV1(
                        strategy_level=strategy_label,
                        outline_steps=["whiteboard", "confirm", "repair" if timing_label != "none" else "observe"],
                        utterance_constraints=["non_diagnostic", "brief", "confirm_before_assume"],
                    ).to_dict(),
                    "care_utterance": CareUtteranceV1(
                        draft_text=f"structured:{strategy_label}",
                        confirmation_question="我刚才的理解贴近你现在的状态吗？",
                        repair_text="如果我理解偏了，你直接纠正我就可以。",
                    ).to_dict(),
                    "scores": {"timing": timing_score, "strategy": strategy_score},
                }
            )
        return output

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(
                {
                    "kind": "structured",
                    "config": self.config,
                    "feature_names": self.feature_names,
                    "timing_model": self.timing_model,
                    "strategy_model": self.strategy_model,
                },
                handle,
            )

    @classmethod
    def load(cls, path: str | Path) -> "StructuredBaseline":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        model = cls(payload["config"])
        model.feature_names = payload.get("feature_names", [])
        model.timing_model = payload["timing_model"]
        model.strategy_model = payload["strategy_model"]
        return model
