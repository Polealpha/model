from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from ..feature_utils import extract_targets
from ..schema import CareUtteranceV1, EpisodeWindowV1, StrategyPlanV1, TimingDecisionV1, STRATEGY_LABELS, TIMING_LABELS
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
class JointConfig(TemporalTrainerConfig):
    architecture: str = "video_temporal_joint"


class JointModel:
    def __init__(self, config: JointConfig | Dict[str, Any] | None = None) -> None:
        if config is None:
            config = JointConfig()
        elif isinstance(config, dict):
            config = JointConfig(**config)
        self.config = config
        self.sequence_feature_names: List[str] = []
        self.template_classes: List[str] = []
        self.model: TemporalMultiHeadModel | None = None
        self.training_summary: Dict[str, Any] = {}

    def _build_dataset(self, episodes: Sequence[EpisodeWindowV1 | Dict[str, Any]]) -> EpisodeTensorDataset:
        static = build_static_arrays(episodes)
        sequence, feature_names = build_temporal_array(episodes, self.config.sequence_steps)
        self.sequence_feature_names = feature_names
        targets = extract_targets(episodes)
        templates = sorted({str(value) for value in targets["template"]})
        if not self.template_classes:
            self.template_classes = templates
        template_map = {name: index for index, name in enumerate(self.template_classes)}
        template_index = np.asarray([template_map[str(value)] for value in targets["template"]], dtype=np.int64)
        return EpisodeTensorDataset(
            sequence=sequence,
            persona=static["persona"],
            context=static["context"],
            observation=static["observation"],
            targets={
                "timing": targets["timing"],
                "strategy": targets["strategy"],
                "template": template_index,
            },
        )

    def fit(self, episodes: Sequence[EpisodeWindowV1 | Dict[str, Any]]) -> "JointModel":
        dataset = self._build_dataset(episodes)
        self.model = TemporalMultiHeadModel(
            sequence_dim=dataset.sequence.shape[-1],
            persona_dim=dataset.persona.shape[-1],
            context_dim=dataset.context.shape[-1],
            observation_dim=dataset.observation.shape[-1],
            hidden_dim=self.config.hidden_dim,
            head_dims={
                "timing": len(TIMING_LABELS),
                "strategy": len(STRATEGY_LABELS),
                "template": len(self.template_classes),
            },
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
        )
        self.training_summary = train_temporal_model(self.model, dataset, self.config, head_names=("timing", "strategy", "template"))
        return self

    def predict(self, episodes: Sequence[EpisodeWindowV1 | Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.model is None:
            raise RuntimeError("JointModel must be fit before prediction")
        dataset = self._build_dataset(episodes)
        outputs = predict_temporal_model(self.model, dataset, self.config, head_names=("timing", "strategy", "template"))
        timing_probs = outputs["timing"]
        strategy_probs = outputs["strategy"]
        template_probs = outputs["template"]
        results: List[Dict[str, Any]] = []
        for idx in range(timing_probs.shape[0]):
            timing_idx = int(np.argmax(timing_probs[idx]))
            strategy_idx = int(np.argmax(strategy_probs[idx]))
            template_idx = int(np.argmax(template_probs[idx]))
            timing_score = float(np.max(timing_probs[idx]))
            strategy_score = float(np.max(strategy_probs[idx]))
            template_name = self.template_classes[template_idx] if self.template_classes else "default"
            strategy_name = STRATEGY_LABELS[strategy_idx]
            results.append(
                {
                    "timing": TimingDecisionV1(
                        decision=TIMING_LABELS[timing_idx],
                        score=timing_score,
                        uncertainty=float(1.0 - timing_score),
                        why_codes=["video_temporal_joint"],
                    ).to_dict(),
                    "strategy": StrategyPlanV1(
                        strategy_level=strategy_name,
                        outline_steps=["observe", "light_guess", "confirm", "repair"],
                        utterance_constraints=["non_diagnostic", "brief", "repairable"],
                    ).to_dict(),
                    "care_utterance": CareUtteranceV1(
                        draft_text=f"{template_name}:{strategy_name}",
                        confirmation_question="我刚才的判断贴近你现在的状态吗？",
                        repair_text="如果我判断偏了，你可以直接纠正我，我会换一种更合适的方式。",
                    ).to_dict(),
                    "scores": {"timing": timing_score, "strategy": strategy_score},
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
                "kind": "joint_temporal",
                "config": asdict(self.config),
                "sequence_feature_names": self.sequence_feature_names,
                "template_classes": self.template_classes,
                "training_summary": self.training_summary,
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "JointModel":
        payload = load_temporal_checkpoint(path)
        metadata = payload["metadata"]
        model = cls(metadata["config"])
        model.sequence_feature_names = metadata.get("sequence_feature_names", [])
        model.template_classes = metadata.get("template_classes", [])
        model.training_summary = metadata.get("training_summary", {})
        model.model = TemporalMultiHeadModel(
            sequence_dim=len(model.sequence_feature_names),
            persona_dim=8,
            context_dim=10,
            observation_dim=13,
            hidden_dim=model.config.hidden_dim,
            head_dims={
                "timing": len(TIMING_LABELS),
                "strategy": len(STRATEGY_LABELS),
                "template": len(model.template_classes),
            },
            num_layers=model.config.num_layers,
            num_heads=model.config.num_heads,
            dropout=model.config.dropout,
        )
        model.model.load_state_dict(payload["state_dict"])
        return model
