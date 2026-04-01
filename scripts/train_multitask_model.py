from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, mean_absolute_error


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.config import load_config
from training.data import load_episodes
from training.feature_utils import extract_targets
from training.io import save_json
from training.multitask.model import MultiTaskModel
from training.schema import STATE_LABELS
from training.synthetic import write_synthetic_dataset
from training.weaklabel import write_weaklabel_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the temporal multitask state + timing model.")
    parser.add_argument("--config", default=REPO_ROOT / "training" / "configs" / "default.yaml")
    parser.add_argument("--synthetic-root", default=REPO_ROOT / "datasets" / "synthetic")
    parser.add_argument("--weaklabel-root", default=REPO_ROOT / "datasets" / "weaklabel")
    parser.add_argument("--weaklabel-db", default=Path(r"E:\Desktop\emotion engine\backend\auth.db"))
    parser.add_argument("--use-weaklabel", action="store_true")
    parser.add_argument("--model-out", default=REPO_ROOT / "training" / "artifacts" / "multitask_model.pt")
    parser.add_argument("--metrics-out", default=REPO_ROOT / "reports" / "multitask_metrics.json")
    return parser.parse_args()


def ensure_synthetic(cfg: dict, data_root: Path) -> None:
    if (data_root / "train.jsonl").exists() and (data_root / "dev.jsonl").exists() and (data_root / "test.jsonl").exists():
        return
    synthetic_cfg = dict(cfg.get("synthetic", {}))
    counts = {
        "train": int(synthetic_cfg.get("train_size", 1200)),
        "dev": int(synthetic_cfg.get("dev_size", 240)),
        "test": int(synthetic_cfg.get("test_size", 240)),
    }
    write_synthetic_dataset(data_root, synthetic_cfg | {"seed": int(cfg.get("random_seed", 42))}, counts)


def ensure_weaklabel(data_root: Path, db_path: Path) -> None:
    if (data_root / "train.jsonl").exists() and (data_root / "dev.jsonl").exists() and (data_root / "test.jsonl").exists():
        return
    write_weaklabel_dataset(db_path, data_root)


def eval_split(model: MultiTaskModel, episodes) -> dict:
    preds = model.predict(episodes)
    targets = extract_targets(episodes)
    timing_gold = targets["timing"].tolist()
    timing_pred = [0 if item["timing"]["decision"] == "none" else 1 if item["timing"]["decision"] == "delay" else 2 for item in preds]
    state_pred = [item["state"] for item in preds]
    state_gold = targets["state"]
    report = classification_report(
        timing_gold,
        timing_pred,
        labels=[0, 1, 2],
        target_names=["none", "delay", "immediate"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "timing_accuracy": float(accuracy_score(timing_gold, timing_pred)),
        "timing_balanced_accuracy": float(balanced_accuracy_score(timing_gold, timing_pred)),
        "timing_macro_f1": float(f1_score(timing_gold, timing_pred, average="macro")),
        "state_mae": float(mean_absolute_error(state_gold, [[row[name] for name in STATE_LABELS] for row in state_pred])),
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


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    synthetic_root = Path(args.synthetic_root)
    weaklabel_root = Path(args.weaklabel_root)
    ensure_synthetic(cfg, synthetic_root)
    if args.use_weaklabel:
        ensure_weaklabel(weaklabel_root, Path(args.weaklabel_db))

    synthetic = {split: load_episodes(synthetic_root / f"{split}.jsonl") for split in ("train", "dev", "test")}
    weaklabel = {split: load_episodes(weaklabel_root / f"{split}.jsonl") for split in ("train", "dev", "test")} if args.use_weaklabel else {"train": [], "dev": [], "test": []}

    train_episodes = synthetic["train"] + weaklabel["train"]
    dev_episodes = synthetic["dev"] + weaklabel["dev"]
    test_episodes = synthetic["test"] + weaklabel["test"]

    model = MultiTaskModel(cfg.get("multitask", {}))
    model.fit(train_episodes)

    metrics = {
        "train": eval_split(model, train_episodes),
        "dev": eval_split(model, dev_episodes),
        "test": eval_split(model, test_episodes),
        "synthetic_test": eval_split(model, synthetic["test"]),
    }
    if args.use_weaklabel and weaklabel["test"]:
        metrics["weaklabel_test"] = eval_split(model, weaklabel["test"])

    model.save(args.model_out)
    payload = {
        "model": "multitask_temporal",
        "metrics": metrics,
        "config": cfg.get("multitask", {}),
        "train_sources": {
            "synthetic_train": len(synthetic["train"]),
            "weaklabel_train": len(weaklabel["train"]),
        },
    }
    save_json(args.metrics_out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
