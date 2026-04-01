from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, mean_absolute_error


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.baselines.structured import StructuredBaseline
from training.data import load_episodes
from training.feature_utils import extract_targets
from training.joint.model import JointModel
from training.multitask.model import MultiTaskModel
from training.schema import STATE_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained timing model on a dataset split.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def load_model(path: str | Path):
    path = Path(path)
    if path.suffix == ".pkl":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict) and payload.get("kind") == "structured":
            return StructuredBaseline.load(path), "structured"
        raise ValueError(f"Unknown pickle artifact: {path}")
    payload = torch.load(path, map_location="cpu")
    metadata = payload.get("metadata", {})
    kind = metadata.get("kind")
    if kind == "multitask_temporal":
        return MultiTaskModel.load(path), kind
    if kind == "joint_temporal":
        return JointModel.load(path), kind
    raise ValueError(f"Unknown model artifact: {path}")


def evaluate(model, kind: str, episodes) -> Dict[str, Any]:
    preds = model.predict(episodes)
    targets = extract_targets(episodes)
    timing_gold = targets["timing"].tolist()
    timing_pred = [0 if item["timing"]["decision"] == "none" else 1 if item["timing"]["decision"] == "delay" else 2 for item in preds]
    report = classification_report(
        timing_gold,
        timing_pred,
        labels=[0, 1, 2],
        target_names=["none", "delay", "immediate"],
        output_dict=True,
        zero_division=0,
    )
    metrics: Dict[str, Any] = {
        "timing_accuracy": float(accuracy_score(timing_gold, timing_pred)),
        "timing_balanced_accuracy": float(balanced_accuracy_score(timing_gold, timing_pred)),
        "timing_macro_f1": float(f1_score(timing_gold, timing_pred, average="macro")),
        "timing_per_class": {
            key: {
                "precision": float(value["precision"]),
                "recall": float(value["recall"]),
                "f1": float(value["f1-score"]),
                "support": float(value["support"]),
            }
            for key, value in report.items()
            if key in {"none", "delay", "immediate"}
        },
    }
    if any("strategy" in item for item in preds):
        strategy_gold = targets["strategy"].tolist()
        strategy_pred = [0 if item["strategy"]["strategy_level"] == "observe" else 1 if item["strategy"]["strategy_level"] == "nudge" else 2 if item["strategy"]["strategy_level"] == "care" else 3 for item in preds]
        metrics["strategy_accuracy"] = float(accuracy_score(strategy_gold, strategy_pred))
        metrics["strategy_macro_f1"] = float(f1_score(strategy_gold, strategy_pred, average="macro"))
    if any("state" in item for item in preds):
        state_pred = np.array([[row["state"][name] for name in STATE_LABELS] for row in preds], dtype=np.float32)
        state_gold = targets["state"]
        metrics["state_mae"] = float(mean_absolute_error(state_gold, state_pred))
    metrics["artifact_kind"] = kind
    return metrics


def main() -> int:
    args = parse_args()
    episodes = load_episodes(args.data_path)
    model, kind = load_model(args.model_path)
    metrics = evaluate(model, kind, episodes)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
