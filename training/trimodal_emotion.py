from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score, mean_absolute_error
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .io import load_json, save_json


FINE_EMOTION_LABELS = (
    "neutral",
    "calm",
    "content",
    "joy",
    "relief",
    "tired",
    "sadness",
    "lonely",
    "anxiety",
    "frustration",
    "anger",
    "overwhelmed",
)

STATE_TARGETS = ("stress", "fatigue", "attention_drop", "suppression")

EMOTION_PROTOTYPES: Dict[str, Dict[str, float]] = {
    "neutral": {"valence": 0.54, "arousal": 0.42, "stress": 0.25, "fatigue": 0.32, "attention_drop": 0.28, "suppression": 0.22},
    "calm": {"valence": 0.68, "arousal": 0.24, "stress": 0.16, "fatigue": 0.22, "attention_drop": 0.18, "suppression": 0.18},
    "content": {"valence": 0.76, "arousal": 0.36, "stress": 0.14, "fatigue": 0.20, "attention_drop": 0.18, "suppression": 0.16},
    "joy": {"valence": 0.90, "arousal": 0.80, "stress": 0.18, "fatigue": 0.14, "attention_drop": 0.10, "suppression": 0.08},
    "relief": {"valence": 0.72, "arousal": 0.48, "stress": 0.22, "fatigue": 0.18, "attention_drop": 0.16, "suppression": 0.14},
    "tired": {"valence": 0.40, "arousal": 0.22, "stress": 0.28, "fatigue": 0.82, "attention_drop": 0.76, "suppression": 0.38},
    "sadness": {"valence": 0.18, "arousal": 0.34, "stress": 0.52, "fatigue": 0.44, "attention_drop": 0.46, "suppression": 0.60},
    "lonely": {"valence": 0.24, "arousal": 0.28, "stress": 0.48, "fatigue": 0.40, "attention_drop": 0.34, "suppression": 0.72},
    "anxiety": {"valence": 0.16, "arousal": 0.84, "stress": 0.88, "fatigue": 0.38, "attention_drop": 0.58, "suppression": 0.34},
    "frustration": {"valence": 0.20, "arousal": 0.70, "stress": 0.72, "fatigue": 0.34, "attention_drop": 0.42, "suppression": 0.30},
    "anger": {"valence": 0.10, "arousal": 0.90, "stress": 0.84, "fatigue": 0.28, "attention_drop": 0.26, "suppression": 0.22},
    "overwhelmed": {"valence": 0.08, "arousal": 0.78, "stress": 0.92, "fatigue": 0.74, "attention_drop": 0.80, "suppression": 0.48},
}


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _clip01_array(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(np.float32), 0.0, 1.0)


def _stable_label(index: int) -> str:
    return FINE_EMOTION_LABELS[index % len(FINE_EMOTION_LABELS)]


def _sequence_curve(kind: str, steps: int) -> np.ndarray:
    x = np.linspace(0.0, 1.0, num=steps, dtype=np.float32)
    if kind == "rising":
        return x
    if kind == "falling":
        return 1.0 - x
    if kind == "spike":
        return np.exp(-((x - 0.72) ** 2) / 0.015).astype(np.float32)
    if kind == "dip":
        return np.exp(-((x - 0.35) ** 2) / 0.03).astype(np.float32)
    if kind == "oscillate":
        return (0.5 + 0.5 * np.sin(2.0 * np.pi * (1.5 * x + 0.15))).astype(np.float32)
    return np.ones_like(x, dtype=np.float32) * 0.5


@dataclass
class TriModalSyntheticConfig:
    seed: int = 42
    sequence_steps: int = 48
    face_dim: int = 10
    motion_dim: int = 10
    audio_dim: int = 10
    persona_dim: int = 6


class TriModalSyntheticGenerator:
    def __init__(self, cfg: TriModalSyntheticConfig | Dict[str, Any] | None = None) -> None:
        if cfg is None:
            cfg = TriModalSyntheticConfig()
        elif isinstance(cfg, dict):
            allowed = {
                "seed",
                "sequence_steps",
                "face_dim",
                "motion_dim",
                "audio_dim",
                "persona_dim",
            }
            cfg = TriModalSyntheticConfig(**{key: value for key, value in cfg.items() if key in allowed})
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def generate_split(self, total: int, split_name: str) -> Dict[str, np.ndarray]:
        face = np.zeros((total, self.cfg.sequence_steps, self.cfg.face_dim), dtype=np.float32)
        motion = np.zeros((total, self.cfg.sequence_steps, self.cfg.motion_dim), dtype=np.float32)
        audio = np.zeros((total, self.cfg.sequence_steps, self.cfg.audio_dim), dtype=np.float32)
        persona = np.zeros((total, self.cfg.persona_dim), dtype=np.float32)
        emotion_idx = np.zeros((total,), dtype=np.int64)
        valence_arousal = np.zeros((total, 2), dtype=np.float32)
        states = np.zeros((total, len(STATE_TARGETS)), dtype=np.float32)
        support_need = np.zeros((total,), dtype=np.float32)

        for index in range(total):
            sample = self._sample(index=index, split_name=split_name)
            face[index] = sample["face"]
            motion[index] = sample["motion"]
            audio[index] = sample["audio"]
            persona[index] = sample["persona"]
            emotion_idx[index] = sample["emotion_idx"]
            valence_arousal[index] = sample["va"]
            states[index] = sample["state"]
            support_need[index] = sample["support_need"]

        return {
            "face": face,
            "motion": motion,
            "audio": audio,
            "persona": persona,
            "emotion": emotion_idx,
            "va": valence_arousal,
            "state": states,
            "support_need": support_need,
        }

    def _sample(self, index: int, split_name: str) -> Dict[str, np.ndarray]:
        emotion_name = _stable_label(index)
        emotion_idx = FINE_EMOTION_LABELS.index(emotion_name)
        proto = EMOTION_PROTOTYPES[emotion_name]
        persona = self._sample_persona()
        va, states, support_need = self._sample_targets(emotion_name, proto, persona)
        face, motion, audio = self._sample_modal_sequences(emotion_name, va, states, persona)
        return {
            "face": face,
            "motion": motion,
            "audio": audio,
            "persona": persona.astype(np.float32),
            "emotion_idx": np.asarray(emotion_idx, dtype=np.int64),
            "va": np.asarray(va, dtype=np.float32),
            "state": np.asarray(states, dtype=np.float32),
            "support_need": np.asarray(support_need, dtype=np.float32),
        }

    def _sample_persona(self) -> np.ndarray:
        extraversion = float(self.rng.beta(2.0, 2.0))
        neuroticism = float(self.rng.beta(2.2, 1.8))
        self_reliance = float(self.rng.beta(2.0, 2.2))
        help_seeking = _clamp01(1.0 - 0.55 * self_reliance + 0.20 * extraversion + 0.10 * self.rng.normal())
        stress_sensitivity = _clamp01(0.25 + 0.55 * neuroticism + 0.05 * self.rng.normal())
        recovery_speed = _clamp01(0.65 - 0.35 * neuroticism + 0.18 * extraversion + 0.08 * self.rng.normal())
        return np.asarray(
            [
                extraversion,
                neuroticism,
                self_reliance,
                help_seeking,
                stress_sensitivity,
                recovery_speed,
            ],
            dtype=np.float32,
        )

    def _sample_targets(self, emotion_name: str, proto: Mapping[str, float], persona: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        extraversion, neuroticism, self_reliance, help_seeking, stress_sensitivity, recovery_speed = persona.tolist()
        valence = _clamp01(proto["valence"] + 0.06 * self.rng.normal())
        arousal = _clamp01(proto["arousal"] + 0.06 * self.rng.normal())
        stress = _clamp01(proto["stress"] + 0.14 * stress_sensitivity - 0.08 * recovery_speed + 0.05 * self.rng.normal())
        fatigue = _clamp01(proto["fatigue"] + 0.08 * (1.0 - recovery_speed) + 0.05 * self.rng.normal())
        attention_drop = _clamp01(proto["attention_drop"] + 0.06 * fatigue + 0.05 * self.rng.normal())
        suppression = _clamp01(proto["suppression"] + 0.12 * self_reliance - 0.08 * help_seeking + 0.05 * self.rng.normal())

        if emotion_name in {"joy", "relief", "content", "calm"}:
            support_need = _clamp01(0.08 + 0.12 * (1.0 - self_reliance) + 0.05 * self.rng.normal())
        else:
            support_need = _clamp01(0.20 + 0.35 * help_seeking + 0.20 * stress + 0.10 * fatigue + 0.08 * attention_drop - 0.08 * suppression + 0.06 * self.rng.normal())

        va = np.asarray([valence, arousal], dtype=np.float32)
        state = np.asarray([stress, fatigue, attention_drop, suppression], dtype=np.float32)
        return va, state, float(support_need)

    def _sample_modal_sequences(
        self,
        emotion_name: str,
        va: np.ndarray,
        state: np.ndarray,
        persona: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        valence, arousal = va.tolist()
        stress, fatigue, attention_drop, suppression = state.tolist()
        extraversion, neuroticism, self_reliance, help_seeking, stress_sensitivity, recovery_speed = persona.tolist()
        steps = self.cfg.sequence_steps
        profile = {
            "joy": "oscillate",
            "relief": "falling",
            "calm": "stable",
            "content": "stable",
            "tired": "falling",
            "sadness": "dip",
            "lonely": "stable",
            "anxiety": "rising",
            "frustration": "oscillate",
            "anger": "spike",
            "overwhelmed": "rising",
        }.get(emotion_name, "stable")
        curve = _sequence_curve(profile, steps)
        curve2 = _sequence_curve("oscillate", steps)
        noise = lambda dim, scale=0.035: self.rng.normal(0.0, scale, size=(steps, dim)).astype(np.float32)

        face = np.stack(
            [
                np.ones(steps, dtype=np.float32) * _clamp01(0.40 + 0.55 * max(valence - 0.5, 0.0)),  # smile
                _clip01_array(0.18 + 0.60 * stress * curve),  # brow furrow
                np.ones(steps, dtype=np.float32) * _clamp01(0.15 + 0.55 * attention_drop),  # eye closure
                np.ones(steps, dtype=np.float32) * _clamp01(0.12 + 0.55 * suppression),  # lip press
                np.ones(steps, dtype=np.float32) * _clamp01(0.20 + 0.45 * arousal),  # jaw drop
                np.ones(steps, dtype=np.float32) * _clamp01(0.18 + 0.50 * attention_drop + 0.10 * suppression),  # gaze avert
                _clip01_array(0.20 + 0.55 * neuroticism * curve2),  # blink irregularity
                _clip01_array(0.20 + 0.48 * arousal * curve),  # head tremor
                np.ones(steps, dtype=np.float32) * _clamp01(0.18 + 0.45 * max(0.5 - valence, 0.0)),  # cheek tension
                np.ones(steps, dtype=np.float32) * _clamp01(0.16 + 0.45 * fatigue),  # face flatness
            ],
            axis=-1,
        ).astype(np.float32)
        face += noise(self.cfg.face_dim)
        face = np.clip(face, 0.0, 1.0)

        motion = np.stack(
            [
                np.ones(steps, dtype=np.float32) * _clamp01(0.18 + 0.72 * fatigue),  # posture slouch
                _clip01_array(0.20 + 0.55 * arousal * curve),  # head speed
                _clip01_array(0.10 + 0.68 * stress * curve2),  # fidget
                np.ones(steps, dtype=np.float32) * _clamp01(0.12 + 0.60 * fatigue),  # shoulder drop
                np.ones(steps, dtype=np.float32) * _clamp01(0.08 + 0.62 * attention_drop),  # freeze
                _clip01_array(0.12 + 0.52 * arousal * curve),  # agitation
                np.ones(steps, dtype=np.float32) * _clamp01(0.15 + 0.48 * help_seeking * max(0.5 - valence, 0.0)),  # reach out
                np.ones(steps, dtype=np.float32) * _clamp01(0.18 + 0.42 * suppression),  # guarded posture
                np.ones(steps, dtype=np.float32) * _clamp01(0.14 + 0.46 * max(valence - 0.5, 0.0) * extraversion),  # open posture
                np.ones(steps, dtype=np.float32) * _clamp01(0.10 + 0.60 * attention_drop + 0.15 * fatigue),  # desk slump
            ],
            axis=-1,
        ).astype(np.float32)
        motion += noise(self.cfg.motion_dim)
        motion = np.clip(motion, 0.0, 1.0)

        audio = np.stack(
            [
                _clip01_array(0.28 + 0.52 * arousal * curve),  # energy
                np.ones(steps, dtype=np.float32) * _clamp01(0.24 + 0.50 * arousal),  # pitch
                _clip01_array(0.18 + 0.58 * stress * curve2),  # pitch var
                np.ones(steps, dtype=np.float32) * _clamp01(0.26 + 0.42 * extraversion - 0.30 * fatigue),  # speech rate
                np.ones(steps, dtype=np.float32) * _clamp01(0.12 + 0.65 * stress),  # jitter
                np.ones(steps, dtype=np.float32) * _clamp01(0.12 + 0.55 * stress_sensitivity),  # shimmer
                np.ones(steps, dtype=np.float32) * _clamp01(0.18 + 0.52 * fatigue + 0.12 * suppression),  # pause ratio
                np.ones(steps, dtype=np.float32) * _clamp01(0.22 + 0.50 * max(valence - 0.5, 0.0)),  # brightness
                np.ones(steps, dtype=np.float32) * _clamp01(0.16 + 0.56 * suppression),  # breathiness
                np.ones(steps, dtype=np.float32) * _clamp01(0.14 + 0.44 * max(0.5 - valence, 0.0) + 0.20 * self_reliance),  # flat prosody
            ],
            axis=-1,
        ).astype(np.float32)
        audio += noise(self.cfg.audio_dim)
        audio = np.clip(audio, 0.0, 1.0)
        return face, motion, audio


def write_trimodal_dataset(root: str | Path, cfg: Mapping[str, Any], counts: Mapping[str, int]) -> Dict[str, str]:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    generator = TriModalSyntheticGenerator(dict(cfg))
    outputs: Dict[str, str] = {}
    manifest = {
        "config": dict(cfg),
        "counts": {key: int(value) for key, value in counts.items()},
        "fine_emotion_labels": list(FINE_EMOTION_LABELS),
        "state_targets": list(STATE_TARGETS),
    }
    for split, count in counts.items():
        payload = generator.generate_split(int(count), split)
        target = root_path / f"{split}.npz"
        np.savez_compressed(target, **payload)
        outputs[split] = str(target)
    save_json(root_path / "manifest.json", manifest)
    return outputs


class TriModalDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        payload = np.load(Path(path))
        self.face = torch.from_numpy(payload["face"]).float()
        self.motion = torch.from_numpy(payload["motion"]).float()
        self.audio = torch.from_numpy(payload["audio"]).float()
        self.persona = torch.from_numpy(payload["persona"]).float()
        self.emotion = torch.from_numpy(payload["emotion"]).long()
        self.va = torch.from_numpy(payload["va"]).float()
        self.state = torch.from_numpy(payload["state"]).float()
        self.support_need = torch.from_numpy(payload["support_need"]).float()

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
        }


class SequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(sequence)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.transformer(x)
        x = self.norm(x)
        return torch.cat([x.mean(dim=1), torch.max(x, dim=1).values], dim=-1)


class TriModalEmotionModel(nn.Module):
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
        self.emotion_head = nn.Linear(hidden_dim, len(FINE_EMOTION_LABELS))
        self.va_head = nn.Linear(hidden_dim, 2)
        self.state_head = nn.Linear(hidden_dim, len(STATE_TARGETS))
        self.support_head = nn.Linear(hidden_dim, 1)

    def forward(self, face: torch.Tensor, motion: torch.Tensor, audio: torch.Tensor, persona: torch.Tensor) -> Dict[str, torch.Tensor]:
        face_feat = self.face_encoder(face)
        motion_feat = self.motion_encoder(motion)
        audio_feat = self.audio_encoder(audio)
        persona_feat = self.persona_encoder(persona)
        fused = torch.cat(
            [
                face_feat,
                motion_feat,
                audio_feat,
                persona_feat,
            ],
            dim=-1,
        )
        shared = self.fusion(fused)
        return {
            "emotion": self.emotion_head(shared),
            "va": self.va_head(shared),
            "state": self.state_head(shared),
            "support_need": self.support_head(shared).squeeze(-1),
        }


@dataclass
class TriModalTrainConfig:
    batch_size: int = 512
    hidden_dim: int = 384
    epochs: int = 24
    lr: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    num_layers: int = 4
    num_heads: int = 8
    label_smoothing: float = 0.05
    va_loss_weight: float = 0.6
    state_loss_weight: float = 0.4
    support_loss_weight: float = 0.2
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


def train_trimodal_model(model: TriModalEmotionModel, dataset: TriModalDataset, cfg: TriModalTrainConfig) -> Dict[str, Any]:
    _set_seed(cfg.seed)
    device = _resolve_device(cfg.device)
    model.to(device)
    runtime_model = _parallelize(model, device, cfg.multi_gpu)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(runtime_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    cls_loss = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    reg_loss = nn.SmoothL1Loss()
    support_loss = nn.MSELoss()
    history: List[float] = []
    for _epoch in range(cfg.epochs):
        runtime_model.train()
        running = 0.0
        total = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = runtime_model(batch["face"], batch["motion"], batch["audio"], batch["persona"])
            loss = cls_loss(outputs["emotion"], batch["emotion"])
            loss = loss + cfg.va_loss_weight * reg_loss(outputs["va"], batch["va"])
            loss = loss + cfg.state_loss_weight * reg_loss(outputs["state"], batch["state"])
            loss = loss + cfg.support_loss_weight * support_loss(outputs["support_need"], batch["support_need"])
            loss.backward()
            nn.utils.clip_grad_norm_(runtime_model.parameters(), cfg.grad_clip)
            optimizer.step()
            running += float(loss.item()) * int(batch["emotion"].shape[0])
            total += int(batch["emotion"].shape[0])
        history.append(running / max(1, total))
    return {"loss_history": history, "device": str(device), "device_count": int(torch.cuda.device_count() if device.type == "cuda" else 0)}


@torch.no_grad()
def predict_trimodal_model(
    model: TriModalEmotionModel,
    dataset: TriModalDataset,
    cfg: TriModalTrainConfig,
    mode: str = "full",
) -> Dict[str, np.ndarray]:
    device = _resolve_device(cfg.device)
    model.to(device)
    runtime_model = _parallelize(model, device, cfg.multi_gpu)
    runtime_model.eval()
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    outputs: Dict[str, List[np.ndarray]] = {"emotion": [], "va": [], "state": [], "support_need": []}
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        modal = _modality_batch(batch, mode)
        pred = runtime_model(modal["face"], modal["motion"], modal["audio"], modal["persona"])
        outputs["emotion"].append(torch.softmax(pred["emotion"], dim=-1).cpu().numpy())
        outputs["va"].append(pred["va"].cpu().numpy())
        outputs["state"].append(pred["state"].cpu().numpy())
        outputs["support_need"].append(pred["support_need"].cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in outputs.items()}


def evaluate_trimodal_model(
    model: TriModalEmotionModel,
    dataset: TriModalDataset,
    cfg: TriModalTrainConfig,
    mode: str = "full",
) -> Dict[str, Any]:
    pred = predict_trimodal_model(model, dataset, cfg, mode=mode)
    emotion_pred = np.argmax(pred["emotion"], axis=-1)
    emotion_gold = dataset.emotion.numpy()
    report = classification_report(
        emotion_gold,
        emotion_pred,
        labels=list(range(len(FINE_EMOTION_LABELS))),
        target_names=list(FINE_EMOTION_LABELS),
        output_dict=True,
        zero_division=0,
    )
    return {
        "emotion_accuracy": float(accuracy_score(emotion_gold, emotion_pred)),
        "emotion_balanced_accuracy": float(balanced_accuracy_score(emotion_gold, emotion_pred)),
        "emotion_macro_f1": float(f1_score(emotion_gold, emotion_pred, average="macro")),
        "valence_mae": float(mean_absolute_error(dataset.va.numpy()[:, 0], pred["va"][:, 0])),
        "arousal_mae": float(mean_absolute_error(dataset.va.numpy()[:, 1], pred["va"][:, 1])),
        "state_mae": float(mean_absolute_error(dataset.state.numpy(), pred["state"])),
        "support_need_mae": float(mean_absolute_error(dataset.support_need.numpy(), pred["support_need"])),
        "per_class": {
            name: {
                "precision": float(value["precision"]),
                "recall": float(value["recall"]),
                "f1": float(value["f1-score"]),
                "support": float(value["support"]),
            }
            for name, value in report.items()
            if name in set(FINE_EMOTION_LABELS)
        },
    }


def save_trimodal_checkpoint(path: str | Path, model: TriModalEmotionModel, metadata: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": dict(metadata)}, target)


def load_trimodal_checkpoint(path: str | Path) -> Dict[str, Any]:
    return torch.load(Path(path), map_location="cpu")


def manifest_labels(root: str | Path) -> Dict[str, Any]:
    return load_json(Path(root) / "manifest.json")


def train_config_dict(cfg: TriModalTrainConfig) -> Dict[str, Any]:
    return asdict(cfg)
