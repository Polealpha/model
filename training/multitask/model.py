from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from ..feature_utils import extract_targets
from ..schema import EpisodeWindowV1, StateVectorV1, TimingDecisionV1, TIMING_LABELS
from ..temporal_backbone import (
    EpisodeTensorDataset,
    TemporalMultiHeadModel,
    TemporalTrainerConfig,
    build_static_arrays,
    build_temporal_array,
    load_temporal_checkpoint,
    predict_temporal_model,
    save_temporal_checkpoint,
    train_temporal_model,
)


@dataclass
class MultiTaskConfig(TemporalTrainerConfig):
    architecture: str = "video_temporal"


class MultiTaskModel:
    def __init__(self, config: MultiTaskConfig | Dict[str, Any] | None = None) -> None:
        if config is None:
            config = MultiTaskConfig()
        elif isinstance(config, dict):
            config = MultiTaskConfig(**config)
        self.config = config
        self.sequence_feature_names: List[str] = []
        self.model: TemporalMultiHeadModel | None = None
        self.training_summary: Dict[str, Any] = {}

    def _build_dataset(self, episodes: Sequence[EpisodeWindowV1 | Dict[str, Any]]) -> EpisodeTensorDataset:
        static = build_static_arrays(episodes)
        sequence, feature_names = build_temporal_array(episodes, self.config.sequence_steps)
        self.sequence_feature_names = feature_names
        targets = extract_targets(episodes)
        return EpisodeTensorDataset(
            sequence=sequence,
            persona=static["persona"],
            context=static["context"],
            observation=static["observation"],
            targets={"timing": targets["timing"], "state": targets["state"]},
        )

    def fit(self, episodes: Sequence[EpisodeWindowV1 | Dict[str, Any]]) -> "MultiTaskModel":
        dataset = self._build_dataset(episodes)
        self.model = TemporalMultiHeadModel(
            sequence_dim=dataset.sequence.shape[-1],
            persona_dim=dataset.persona.shape[-1],
            context_dim=dataset.context.shape[-1],
            observation_dim=dataset.observation.shape[-1],
            hidden_dim=self.config.hidden_dim,
            head_dims={"timing": len(TIMING_LABELS), "state": 6},
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
        )
        self.training_summary = train_temporal_model(self.model, dataset, self.config, head_names=("timing", "state"))
        return self

    def predict(self, episodes: Sequence[EpisodeWindowV1 | Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("MultiTaskModel must be fit before prediction")
        dataset = self._build_dataset(episodes)
        outputs = predict_temporal_model(self.model, dataset, self.config, head_names=("timing", "state"))
        timing_probs = outputs["timing"]
        state_pred = outputs["state"]
        results: List[Dict[str, Any]] = []
        for idx in range(timing_probs.shape[0]):
            timing_idx = int(np.argmax(timing_probs[idx]))
            timing_score = float(np.max(timing_probs[idx]))
            state_values = np.clip(state_pred[idx], 0.0, 1.0)
            results.append(
                {
                    "timing": TimingDecisionV1(
                        decision=TIMING_LABELS[timing_idx],
                        score=timing_score,
                        uncertainty=float(1.0 - timing_score),
                        why_codes=["video_temporal_backbone"],
                    ).to_dict(),
                    "state": StateVectorV1(
                        valence=float(state_values[0]),
                        arousal=float(state_values[1]),
                        stress=float(state_values[2]),
                        fatigue=float(state_values[3]),
                        attention_drop=float(state_values[4]),
                        suppression=float(state_values[5]),
                    ).to_dict(),
                    "scores": {"timing": timing_score},
                }
            )
        return results

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Cannot save an unfit model")
        save_temporal_checkpoint(
            path,
            self.model,
            metadata={
                "kind": "multitask_temporal",
                "config": asdict(self.config),
                "sequence_feature_names": self.sequence_feature_names,
                "training_summary": self.training_summary,
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "MultiTaskModel":
        payload = load_temporal_checkpoint(path)
        metadata = payload["metadata"]
        model = cls(metadata["config"])
        model.sequence_feature_names = metadata.get("sequence_feature_names", [])
        model.training_summary = metadata.get("training_summary", {})
        model.model = TemporalMultiHeadModel(
            sequence_dim=len(model.sequence_feature_names),
            persona_dim=8,
            context_dim=10,
            observation_dim=13,
            hidden_dim=model.config.hidden_dim,
            head_dims={"timing": len(TIMING_LABELS), "state": 6},
            num_layers=model.config.num_layers,
            num_heads=model.config.num_heads,
            dropout=model.config.dropout,
        )
        model.model.load_state_dict(payload["state_dict"])
        return model
