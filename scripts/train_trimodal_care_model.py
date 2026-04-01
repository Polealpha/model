from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.config import load_config
from training.io import save_json
from training.trimodal_care import (
    TriModalCareDataset,
    TriModalCareModel,
    TriModalCareTrainConfig,
    evaluate_trimodal_care_model,
    load_pretrained_encoders,
    save_trimodal_care_checkpoint,
    train_config_dict,
    train_trimodal_care_model,
)
from training.trimodal_emotion import TriModalDataset, manifest_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tri-modal care-need model.")
    parser.add_argument("--config", default=REPO_ROOT / "training" / "configs" / "remote_a800_trimodal_care.yaml")
    parser.add_argument("--data-root", default=REPO_ROOT / "datasets" / "trimodal_emotion")
    parser.add_argument("--pretrained", default=REPO_ROOT / "training" / "artifacts" / "trimodal_emotion_model.pt")
    parser.add_argument("--model-out", default=REPO_ROOT / "training" / "artifacts" / "trimodal_care_model.pt")
    parser.add_argument("--metrics-out", default=REPO_ROOT / "reports" / "trimodal_care_metrics.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = Path(args.data_root)
    train_ds = TriModalCareDataset(TriModalDataset(data_root / "train.npz"))
    dev_ds = TriModalCareDataset(TriModalDataset(data_root / "dev.npz"))
    test_ds = TriModalCareDataset(TriModalDataset(data_root / "test.npz"))

    train_cfg = TriModalCareTrainConfig(**dict(cfg.get("trimodal_care_model", {})))
    model = TriModalCareModel(
        face_dim=train_ds.face.shape[-1],
        motion_dim=train_ds.motion.shape[-1],
        audio_dim=train_ds.audio.shape[-1],
        persona_dim=train_ds.persona.shape[-1],
        hidden_dim=train_cfg.hidden_dim,
        num_layers=train_cfg.num_layers,
        num_heads=train_cfg.num_heads,
        dropout=train_cfg.dropout,
    )
    pretrained_path = Path(args.pretrained)
    if pretrained_path.exists():
        load_pretrained_encoders(model, pretrained_path)
    training_summary = train_trimodal_care_model(model, train_ds, train_cfg)
    metrics = {
        "train": evaluate_trimodal_care_model(model, train_ds, train_cfg, mode="full"),
        "dev": evaluate_trimodal_care_model(model, dev_ds, train_cfg, mode="full"),
        "test": evaluate_trimodal_care_model(model, test_ds, train_cfg, mode="full"),
        "ablations": {
            "face_only": evaluate_trimodal_care_model(model, test_ds, train_cfg, mode="face_only"),
            "motion_only": evaluate_trimodal_care_model(model, test_ds, train_cfg, mode="motion_only"),
            "audio_only": evaluate_trimodal_care_model(model, test_ds, train_cfg, mode="audio_only"),
        },
    }
    save_trimodal_care_checkpoint(
        args.model_out,
        model,
        metadata={
            "kind": "trimodal_care",
            "config": train_config_dict(train_cfg),
            "dataset_manifest": manifest_labels(data_root),
            "training_summary": training_summary,
            "pretrained": str(pretrained_path),
        },
    )
    payload = {
        "model": "trimodal_care",
        "metrics": metrics,
        "config": train_config_dict(train_cfg),
        "dataset_manifest": manifest_labels(data_root),
        "training_summary": training_summary,
        "pretrained": str(pretrained_path),
    }
    save_json(args.metrics_out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
