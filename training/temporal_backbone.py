from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .feature_utils import (
    DEFAULT_CONTEXT_FEATURES,
    DEFAULT_OBSERVATION_FEATURES,
    DEFAULT_PERSONA_FEATURES,
    episode_to_feature_dict,
)
from .schema import EpisodeWindowV1, STATE_LABELS, STRATEGY_LABELS, TIMING_LABELS, episode_from_dict


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _episode_obj(episode: EpisodeWindowV1 | Dict[str, Any]) -> EpisodeWindowV1:
    return episode if isinstance(episode, EpisodeWindowV1) else episode_from_dict(episode)


def _stable_seed(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _temporal_feature_names() -> List[str]:
    return list(DEFAULT_OBSERVATION_FEATURES)


def build_static_arrays(episodes: Sequence[EpisodeWindowV1 | Dict[str, Any]]) -> Dict[str, np.ndarray]:
    persona_rows: List[List[float]] = []
    context_rows: List[List[float]] = []
    obs_rows: List[List[float]] = []
    for raw in episodes:
        episode = _episode_obj(raw)
        feature_dict = episode_to_feature_dict(episode)
        persona_rows.append([float(feature_dict.get(name, 0.0)) for name in DEFAULT_PERSONA_FEATURES])
        context_rows.append([float(feature_dict.get(name, 0.0)) for name in DEFAULT_CONTEXT_FEATURES])
        obs_rows.append([float(feature_dict.get(name, 0.0)) for name in DEFAULT_OBSERVATION_FEATURES])
    return {
        "persona": np.asarray(persona_rows, dtype=np.float32),
        "context": np.asarray(context_rows, dtype=np.float32),
        "observation": np.asarray(obs_rows, dtype=np.float32),
    }


def _fallback_temporal_sequence(episode: EpisodeWindowV1, sequence_steps: int) -> List[Dict[str, float]]:
    rng = np.random.default_rng(_stable_seed(episode.episode_id))
    obs = episode.observations
    state = episode.state_labels
    busy = float(episode.context_flags.get("busy_speaking", 0.0))
    quiet = float(episode.context_flags.get("quiet_mode", 0.0))
    privacy = float(episode.context_flags.get("privacy_on", 0.0))
    steps: List[Dict[str, float]] = []
    denom = max(1, sequence_steps - 1)
    for index in range(sequence_steps):
        phase = index / denom
        fatigue_trend = state.get("fatigue", 0.0) * phase
        stress_wave = state.get("stress", 0.0) * (0.5 + 0.5 * math.sin(math.pi * phase))
        suppression_curve = state.get("suppression", 0.0) * (1.0 - abs(0.5 - phase) * 1.5)
        attention_curve = state.get("attention_drop", 0.0) * phase
        noise = rng.normal(loc=0.0, scale=0.03, size=len(DEFAULT_OBSERVATION_FEATURES))
        row = {
            "face_presence_ratio": _clamp01(obs.get("face_presence_ratio", 0.0) - 0.15 * attention_curve + noise[0]),
            "gaze_avert_ratio": _clamp01(obs.get("gaze_avert_ratio", 0.0) + 0.18 * attention_curve + 0.08 * suppression_curve + noise[1]),
            "head_motion_var": _clamp01(obs.get("head_motion_var", 0.0) + 0.12 * fatigue_trend + 0.08 * stress_wave + noise[2]),
            "posture_slouch_score": _clamp01(obs.get("posture_slouch_score", 0.0) + 0.20 * fatigue_trend + 0.05 * busy + noise[3]),
            "fidget_score": _clamp01(obs.get("fidget_score", 0.0) + 0.10 * stress_wave - 0.06 * quiet + noise[4]),
            "voice_energy": _clamp01(obs.get("voice_energy", 0.0) - 0.16 * fatigue_trend - 0.05 * privacy + noise[5]),
            "speech_rate": _clamp01(obs.get("speech_rate", 0.0) - 0.08 * fatigue_trend + 0.04 * stress_wave + noise[6]),
            "silence_ratio": _clamp01(obs.get("silence_ratio", 0.0) + 0.10 * fatigue_trend + 0.06 * privacy + noise[7]),
            "prosody_stress": _clamp01(obs.get("prosody_stress", 0.0) + 0.16 * stress_wave + 0.05 * busy + noise[8]),
            "attention_drop_proxy": _clamp01(obs.get("attention_drop_proxy", 0.0) + 0.16 * attention_curve + noise[9]),
            "fatigue_proxy": _clamp01(obs.get("fatigue_proxy", 0.0) + 0.18 * fatigue_trend + noise[10]),
            "stress_proxy": _clamp01(obs.get("stress_proxy", 0.0) + 0.18 * stress_wave + noise[11]),
            "receptivity_proxy": _clamp01(obs.get("receptivity_proxy", 0.0) - 0.20 * busy - 0.14 * privacy + 0.10 * (1.0 - phase) + noise[12]),
        }
        steps.append(row)
    return steps


def build_temporal_array(episodes: Sequence[EpisodeWindowV1 | Dict[str, Any]], sequence_steps: int) -> tuple[np.ndarray, List[str]]:
    feature_names = _temporal_feature_names()
    rows: List[np.ndarray] = []
    for raw in episodes:
        episode = _episode_obj(raw)
        temporal = episode.extra.get("temporal_features", [])
        if not isinstance(temporal, list) or not temporal:
            temporal = _fallback_temporal_sequence(episode, sequence_steps)
        if len(temporal) < sequence_steps:
            temporal = list(temporal) + [temporal[-1]] * (sequence_steps - len(temporal))
        temporal = temporal[:sequence_steps]
        rows.append(np.asarray([[float(step.get(name, 0.0)) for name in feature_names] for step in temporal], dtype=np.float32))
    return np.stack(rows, axis=0), feature_names


class EpisodeTensorDataset(Dataset):
    def __init__(
        self,
        sequence: np.ndarray,
        persona: np.ndarray,
        context: np.ndarray,
        observation: np.ndarray,
        targets: Mapping[str, np.ndarray],
    ) -> None:
        self.sequence = torch.from_numpy(sequence).float()
        self.persona = torch.from_numpy(persona).float()
        self.context = torch.from_numpy(context).float()
        self.observation = torch.from_numpy(observation).float()
        self.targets = {
            name: torch.from_numpy(np.asarray(value)).long() if np.asarray(value).dtype.kind in {"i", "u"} else torch.from_numpy(np.asarray(value)).float()
            for name, value in targets.items()
        }

    def __len__(self) -> int:
        return int(self.sequence.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        batch = {
            "sequence": self.sequence[index],
            "persona": self.persona[index],
            "context": self.context[index],
            "observation": self.observation[index],
        }
        for key, value in self.targets.items():
            batch[key] = value[index]
        return batch


class TemporalVideoBackbone(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        persona_dim: int,
        context_dim: int,
        observation_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sequence_proj = nn.Linear(sequence_dim, hidden_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
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
        self.persona_encoder = nn.Sequential(nn.Linear(persona_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.context_encoder = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.obs_encoder = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        persona: torch.Tensor,
        context: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        seq = self.sequence_proj(sequence)
        seq = self.conv(seq.transpose(1, 2)).transpose(1, 2)
        seq = self.transformer(seq)
        seq_mean = seq.mean(dim=1)
        seq_peak = torch.max(seq, dim=1).values
        fused = torch.cat(
            [
                seq_mean,
                seq_peak,
                self.persona_encoder(persona),
                self.context_encoder(context) + self.obs_encoder(observation),
            ],
            dim=-1,
        )
        return self.fusion(fused)


class TemporalMultiHeadModel(nn.Module):
    def __init__(
        self,
        sequence_dim: int,
        persona_dim: int,
        context_dim: int,
        observation_dim: int,
        hidden_dim: int,
        head_dims: Mapping[str, int],
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = TemporalVideoBackbone(
            sequence_dim=sequence_dim,
            persona_dim=persona_dim,
            context_dim=context_dim,
            observation_dim=observation_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.heads = nn.ModuleDict({name: nn.Linear(hidden_dim, dim) for name, dim in head_dims.items()})

    def forward(self, sequence: torch.Tensor, persona: torch.Tensor, context: torch.Tensor, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(sequence, persona, context, observation)
        return {name: head(features) for name, head in self.heads.items()}


@dataclass
class TemporalTrainerConfig:
    sequence_steps: int = 12
    hidden_dim: int = 128
    batch_size: int = 64
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    num_layers: int = 2
    num_heads: int = 4
    seed: int = 42
    timing_loss_weight: float = 1.0
    state_loss_weight: float = 0.4
    strategy_loss_weight: float = 0.8
    template_loss_weight: float = 0.3
    grad_clip: float = 1.0
    device: str | None = None
    multi_gpu: bool = True


def resolve_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parallelize_model(model: nn.Module, device: torch.device, enabled: bool) -> nn.Module:
    if enabled and device.type == "cuda" and torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        return nn.DataParallel(model)
    return model


def _class_weight(indices: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(indices.astype(np.int64), minlength=size)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _head_loss(
    name: str,
    logits: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    config: TemporalTrainerConfig,
    timing_weight: torch.Tensor,
    strategy_weight: torch.Tensor | None,
    template_weight: torch.Tensor | None,
) -> torch.Tensor:
    if name == "timing":
        return nn.functional.cross_entropy(logits, batch[name], weight=timing_weight) * config.timing_loss_weight
    if name == "strategy":
        return nn.functional.cross_entropy(logits, batch[name], weight=strategy_weight) * config.strategy_loss_weight
    if name == "template":
        return nn.functional.cross_entropy(logits, batch[name], weight=template_weight) * config.template_loss_weight
    if name == "state":
        return nn.functional.smooth_l1_loss(logits, batch[name]) * config.state_loss_weight
    raise KeyError(f"Unsupported head: {name}")


def train_temporal_model(
    model: TemporalMultiHeadModel,
    train_data: Dataset,
    config: TemporalTrainerConfig,
    head_names: Sequence[str],
) -> Dict[str, Any]:
    _set_seed(config.seed)
    device = resolve_device(config.device)
    model.to(device)
    runtime_model = _parallelize_model(model, device, config.multi_gpu)
    loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(runtime_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    train_targets = train_data.targets
    timing_weight = _class_weight(train_targets["timing"].numpy(), len(TIMING_LABELS), device)
    strategy_weight = _class_weight(train_targets["strategy"].numpy(), len(STRATEGY_LABELS), device) if "strategy" in train_targets else None
    template_weight = _class_weight(train_targets["template"].numpy(), int(train_targets["template"].max().item()) + 1, device) if "template" in train_targets else None
    history: List[float] = []
    for _epoch in range(config.epochs):
        runtime_model.train()
        running = 0.0
        count = 0
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = runtime_model(batch["sequence"], batch["persona"], batch["context"], batch["observation"])
            loss = torch.zeros((), device=device)
            for name in head_names:
                loss = loss + _head_loss(name, outputs[name], batch, config, timing_weight, strategy_weight, template_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(runtime_model.parameters(), config.grad_clip)
            optimizer.step()
            running += float(loss.item()) * int(batch["sequence"].shape[0])
            count += int(batch["sequence"].shape[0])
        history.append(running / max(1, count))
    return {"loss_history": history, "device": str(device), "device_count": int(torch.cuda.device_count() if device.type == "cuda" else 0)}


@torch.no_grad()
def predict_temporal_model(
    model: TemporalMultiHeadModel,
    dataset: EpisodeTensorDataset,
    config: TemporalTrainerConfig,
    head_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    device = resolve_device(config.device)
    model.to(device)
    runtime_model = _parallelize_model(model, device, config.multi_gpu)
    runtime_model.eval()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    outputs: Dict[str, List[np.ndarray]] = {name: [] for name in head_names}
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = runtime_model(batch["sequence"], batch["persona"], batch["context"], batch["observation"])
        for name in head_names:
            value = logits[name]
            if name in {"timing", "strategy", "template"}:
                value = nn.functional.softmax(value, dim=-1)
            outputs[name].append(value.detach().cpu().numpy())
    return {name: np.concatenate(chunks, axis=0) for name, chunks in outputs.items()}


def save_temporal_checkpoint(
    path: str | Path,
    model: TemporalMultiHeadModel,
    metadata: Mapping[str, Any],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": dict(metadata)}, target)


def load_temporal_checkpoint(path: str | Path) -> Dict[str, Any]:
    return torch.load(Path(path), map_location="cpu")


def trainer_config_dict(config: TemporalTrainerConfig) -> Dict[str, Any]:
    return asdict(config)
