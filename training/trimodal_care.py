from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .io import save_json
from .trimodal_emotion import FINE_EMOTION_LABELS, STATE_TARGETS, SequenceEncoder, TriModalDataset


CARE_TIMING_LABELS = ("none", "delay", "immediate")


def derive_care_timing_label(support_need: float, valence: float, arousal: float, stress: float) -> int:
    if support_need >= 0.72 or (support_need >= 0.58 and arousal >= 0.56) or (stress >= 0.80 and arousal >= 0.62):
        return 2
    if support_need >= 0.45 or (valence <= 0.28 and stress >= 0.55):
        return 1
    return 0


class TriModalCareDataset(Dataset):
    def __init__(self, base: TriModalDataset) -> None:
        self.face = base.face
        self.motion = base.motion
        self.audio = base.audio
        self.persona = base.persona
        self.emotion = base.emotion
        self.va = base.va
        self.state = base.state
        self.support_need = base.support_need

        need_label = (self.support_need.numpy() >= 0.5).astype(np.float32)
        timing_label = np.asarray(
            [
                derive_care_timing_label(float(self.support_need[i]), float(self.va[i][0]), float(self.va[i][1]), float(self.state[i][0]))
                for i in range(len(self.support_need))
            ],
            dtype=np.int64,
        )
        self.need_label = torch.from_numpy(need_label).float()
        self.timing_label = torch.from_numpy(timing_label).long()

    def __len__(self) -> int:
        return int(self.emotion.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "face": self.face[index],
            "motion": self.motion[index],
            "audio": self.audio[index],
            "persona": self.persona[index],
            "emotion": self.emotion[index],
            "va": self.va[index],
            "state": self.state[index],
            "support_need": self.support_need[index],
            "need_label": self.need_label[index],
            "timing_label": self.timing_label[index],
        }


class TriModalCareModel(nn.Module):
    def __init__(
        self,
        face_dim: int,
        motion_dim: int,
        audio_dim: int,
        persona_dim: int,
        hidden_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.face_encoder = SequenceEncoder(face_dim, hidden_dim, num_layers, num_heads, dropout)
        self.motion_encoder = SequenceEncoder(motion_dim, hidden_dim, num_layers, num_heads, dropout)
        self.audio_encoder = SequenceEncoder(audio_dim, hidden_dim, num_layers, num_heads, dropout)
        self.persona_encoder = nn.Sequential(
            nn.Linear(persona_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        fusion_dim = hidden_dim * 7
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.need_head = nn.Linear(hidden_dim, 1)
        self.timing_head = nn.Linear(hidden_dim, len(CARE_TIMING_LABELS))
        self.support_head = nn.Linear(hidden_dim, 1)
        self.emotion_head = nn.Linear(hidden_dim, len(FINE_EMOTION_LABELS))
        self.va_head = nn.Linear(hidden_dim, 2)
        self.state_head = nn.Linear(hidden_dim, len(STATE_TARGETS))

    def forward(self, face: torch.Tensor, motion: torch.Tensor, audio: torch.Tensor, persona: torch.Tensor) -> Dict[str, torch.Tensor]:
        face_feat = self.face_encoder(face)
        motion_feat = self.motion_encoder(motion)
        audio_feat = self.audio_encoder(audio)
        persona_feat = self.persona_encoder(persona)
        fused = torch.cat([face_feat, motion_feat, audio_feat, persona_feat], dim=-1)
        shared = self.fusion(fused)
        return {
            "need_logit": self.need_head(shared).squeeze(-1),
            "timing": self.timing_head(shared),
            "support_need": self.support_head(shared).squeeze(-1),
            "emotion": self.emotion_head(shared),
            "va": self.va_head(shared),
            "state": self.state_head(shared),
        }


@dataclass
class TriModalCareTrainConfig:
    batch_size: int = 512
    hidden_dim: int = 384
    epochs: int = 18
    lr: float = 2e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    num_layers: int = 4
    num_heads: int = 8
    timing_loss_weight: float = 0.8
    support_loss_weight: float = 0.4
    emotion_loss_weight: float = 0.25
    va_loss_weight: float = 0.2
    state_loss_weight: float = 0.2
    grad_clip: float = 1.0
    device: str | None = None
    multi_gpu: bool = True
    seed: int = 42


def _resolve_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parallelize(model: nn.Module, device: torch.device, enabled: bool) -> nn.Module:
    if enabled and device.type == "cuda" and torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        return nn.DataParallel(model)
    return model


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _class_weight(labels: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=size)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _modality_batch(batch: Dict[str, torch.Tensor], mode: str) -> Dict[str, torch.Tensor]:
    face = batch["face"]
    motion = batch["motion"]
    audio = batch["audio"]
    if mode == "face_only":
        motion = torch.zeros_like(motion)
        audio = torch.zeros_like(audio)
    elif mode == "motion_only":
        face = torch.zeros_like(face)
        audio = torch.zeros_like(audio)
    elif mode == "audio_only":
        face = torch.zeros_like(face)
        motion = torch.zeros_like(motion)
    return {"face": face, "motion": motion, "audio": audio, "persona": batch["persona"]}


def load_pretrained_encoders(model: TriModalCareModel, checkpoint_path: str | Path) -> None:
    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    own_state = model.state_dict()
    transferable = {key: value for key, value in state_dict.items() if key in own_state and own_state[key].shape == value.shape}
    own_state.update(transferable)
    model.load_state_dict(own_state)


def train_trimodal_care_model(model: TriModalCareModel, dataset: TriModalCareDataset, cfg: TriModalCareTrainConfig) -> Dict[str, Any]:
    _set_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    model.to(device)
    runtime_model = _parallelize(model, device, cfg.multi_gpu)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(runtime_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    timing_weight = _class_weight(dataset.timing_label.numpy(), len(CARE_TIMING_LABELS), device)
    cls_loss = nn.CrossEntropyLoss()
    timing_loss = nn.CrossEntropyLoss(weight=timing_weight)
    bce_loss = nn.BCEWithLogitsLoss()
    reg_loss = nn.SmoothL1Loss()
    history: List[float] = []
    for _epoch in range(cfg.epochs):
        runtime_model.train()
        running = 0.0
        total = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            pred = runtime_model(batch["face"], batch["motion"], batch["audio"], batch["persona"])
            loss = bce_loss(pred["need_logit"], batch["need_label"])
            loss = loss + cfg.timing_loss_weight * timing_loss(pred["timing"], batch["timing_label"])
            loss = loss + cfg.support_loss_weight * reg_loss(pred["support_need"], batch["support_need"])
            loss = loss + cfg.emotion_loss_weight * cls_loss(pred["emotion"], batch["emotion"])
            loss = loss + cfg.va_loss_weight * reg_loss(pred["va"], batch["va"])
            loss = loss + cfg.state_loss_weight * reg_loss(pred["state"], batch["state"])
            loss.backward()
            nn.utils.clip_grad_norm_(runtime_model.parameters(), cfg.grad_clip)
            optimizer.step()
            running += float(loss.item()) * int(batch["need_label"].shape[0])
            total += int(batch["need_label"].shape[0])
        history.append(running / max(1, total))
    return {"loss_history": history, "device": str(device), "device_count": int(torch.cuda.device_count() if device.type == "cuda" else 0)}


@torch.no_grad()
def predict_trimodal_care_model(
    model: TriModalCareModel,
    dataset: TriModalCareDataset,
    cfg: TriModalCareTrainConfig,
    mode: str = "full",
) -> Dict[str, np.ndarray]:
    device = _resolve_device(cfg.device)
    model.to(device)
    runtime_model = _parallelize(model, device, cfg.multi_gpu)
    runtime_model.eval()
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    outputs: Dict[str, List[np.ndarray]] = {
        "need_prob": [],
        "timing": [],
        "support_need": [],
        "emotion": [],
        "va": [],
        "state": [],
    }
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        modal = _modality_batch(batch, mode)
        pred = runtime_model(modal["face"], modal["motion"], modal["audio"], modal["persona"])
        outputs["need_prob"].append(torch.sigmoid(pred["need_logit"]).cpu().numpy())
        outputs["timing"].append(torch.softmax(pred["timing"], dim=-1).cpu().numpy())
        outputs["support_need"].append(pred["support_need"].cpu().numpy())
        outputs["emotion"].append(torch.softmax(pred["emotion"], dim=-1).cpu().numpy())
        outputs["va"].append(pred["va"].cpu().numpy())
        outputs["state"].append(pred["state"].cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in outputs.items()}


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "care_need_accuracy": float(accuracy_score(y_true, y_pred)),
        "care_need_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "care_need_precision": float(precision),
        "care_need_recall": float(recall),
        "care_need_f1": float(f1),
        "care_need_auroc": float(roc_auc_score(y_true, y_prob)),
    }


def evaluate_trimodal_care_model(
    model: TriModalCareModel,
    dataset: TriModalCareDataset,
    cfg: TriModalCareTrainConfig,
    mode: str = "full",
) -> Dict[str, Any]:
    pred = predict_trimodal_care_model(model, dataset, cfg, mode=mode)
    need_true = dataset.need_label.numpy().astype(np.int64)
    need_prob = pred["need_prob"]
    timing_true = dataset.timing_label.numpy()
    timing_pred = np.argmax(pred["timing"], axis=-1)
    timing_report = classification_report(
        timing_true,
        timing_pred,
        labels=list(range(len(CARE_TIMING_LABELS))),
        target_names=list(CARE_TIMING_LABELS),
        output_dict=True,
        zero_division=0,
    )
    result = _binary_metrics(need_true, need_prob)
    result.update(
        {
            "timing_accuracy": float(accuracy_score(timing_true, timing_pred)),
            "timing_balanced_accuracy": float(balanced_accuracy_score(timing_true, timing_pred)),
            "timing_macro_f1": float(f1_score(timing_true, timing_pred, average="macro")),
            "support_need_mae": float(mean_absolute_error(dataset.support_need.numpy(), pred["support_need"])),
            "valence_mae": float(mean_absolute_error(dataset.va.numpy()[:, 0], pred["va"][:, 0])),
            "arousal_mae": float(mean_absolute_error(dataset.va.numpy()[:, 1], pred["va"][:, 1])),
            "state_mae": float(mean_absolute_error(dataset.state.numpy(), pred["state"])),
            "timing_per_class": {
                name: {
                    "precision": float(value["precision"]),
                    "recall": float(value["recall"]),
                    "f1": float(value["f1-score"]),
                    "support": float(value["support"]),
                }
                for name, value in timing_report.items()
                if name in set(CARE_TIMING_LABELS)
            },
        }
    )
    return result


def save_trimodal_care_checkpoint(path: str | Path, model: TriModalCareModel, metadata: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": dict(metadata)}, target)


def train_config_dict(cfg: TriModalCareTrainConfig) -> Dict[str, Any]:
    return asdict(cfg)
