from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.config import load_config
from training.io import save_json
from training.trimodal_emotion import (
    TriModalDataset,
    TriModalEmotionModel,
    TriModalTrainConfig,
    evaluate_trimodal_model,
    load_trimodal_checkpoint,
    manifest_labels,
    save_trimodal_checkpoint,
    train_config_dict,
    train_trimodal_model,
    write_trimodal_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tri-modal fine-grained emotion model on synthetic data.")
    parser.add_argument("--config", default=REPO_ROOT / "training" / "configs" / "remote_a800_trimodal.yaml")
    parser.add_argument("--data-root", default=REPO_ROOT / "datasets" / "trimodal_emotion")
    parser.add_argument("--model-out", default=REPO_ROOT / "training" / "artifacts" / "trimodal_emotion_model.pt")
    parser.add_argument("--metrics-out", default=REPO_ROOT / "reports" / "trimodal_emotion_metrics.json")
    return parser.parse_args()


def ensure_dataset(cfg: Dict, root: Path) -> None:
    if (root / "manifest.json").exists() and (root / "train.npz").exists() and (root / "dev.npz").exists() and (root / "test.npz").exists():
        return
    data_cfg = dict(cfg.get("trimodal_synthetic", {}))
    counts = {
        "train": int(data_cfg.get("train_size", 120000)),
        "dev": int(data_cfg.get("dev_size", 12000)),
        "test": int(data_cfg.get("test_size", 12000)),
    }
    write_trimodal_dataset(root, data_cfg, counts)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = Path(args.data_root)
    ensure_dataset(cfg, data_root)
    train_ds = TriModalDataset(data_root / "train.npz")
    dev_ds = TriModalDataset(data_root / "dev.npz")
    test_ds = TriModalDataset(data_root / "test.npz")

    train_cfg = TriModalTrainConfig(**dict(cfg.get("trimodal_model", {})))
    model = TriModalEmotionModel(
        face_dim=train_ds.face.shape[-1],
        motion_dim=train_ds.motion.shape[-1],
        audio_dim=train_ds.audio.shape[-1],
        persona_dim=train_ds.persona.shape[-1],
        hidden_dim=train_cfg.hidden_dim,
        num_layers=train_cfg.num_layers,
        num_heads=train_cfg.num_heads,
        dropout=train_cfg.dropout,
    )
    training_summary = train_trimodal_model(model, train_ds, train_cfg)

    metrics = {
        "train": evaluate_trimodal_model(model, train_ds, train_cfg, mode="full"),
        "dev": evaluate_trimodal_model(model, dev_ds, train_cfg, mode="full"),
        "test": evaluate_trimodal_model(model, test_ds, train_cfg, mode="full"),
        "ablations": {
            "face_only": evaluate_trimodal_model(model, test_ds, train_cfg, mode="face_only"),
            "motion_only": evaluate_trimodal_model(model, test_ds, train_cfg, mode="motion_only"),
            "audio_only": evaluate_trimodal_model(model, test_ds, train_cfg, mode="audio_only"),
        },
    }

    save_trimodal_checkpoint(
        args.model_out,
        model,
        metadata={
            "kind": "trimodal_emotion",
            "config": train_config_dict(train_cfg),
            "dataset_manifest": manifest_labels(data_root),
            "training_summary": training_summary,
        },
    )
    payload = {
        "model": "trimodal_emotion",
        "metrics": metrics,
        "config": train_config_dict(train_cfg),
        "dataset_manifest": manifest_labels(data_root),
        "training_summary": training_summary,
    }
    save_json(args.metrics_out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
