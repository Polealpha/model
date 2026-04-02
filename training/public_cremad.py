from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

from .io import save_json
from .trimodal_emotion import EMOTION_PROTOTYPES, FINE_EMOTION_LABELS


CREMAD_TO_FINE = {
    "ANG": "anger",
    "DIS": "frustration",
    "FEA": "anxiety",
    "HAP": "joy",
    "NEU": "neutral",
    "SAD": "sadness",
}


def _clamp01(value: float | np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    return np.clip(arr, 0.0, 1.0)


@dataclass
class CREMADDatasetConfig:
    repo_root: str
    output_root: str
    sequence_steps: int = 48
    face_dim: int = 10
    motion_dim: int = 10
    audio_dim: int = 10
    persona_dim: int = 6
    seed: int = 42


def _uniform_indices(length: int, steps: int) -> np.ndarray:
    if length <= 1:
        return np.zeros((steps,), dtype=np.int64)
    return np.linspace(0, length - 1, num=steps).astype(np.int64)


def _iter_frames(video_path: Path) -> List[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    frames: List[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
    finally:
        cap.release()
    return frames


def _center_crop(gray: np.ndarray, ratio: float = 0.78) -> np.ndarray:
    h, w = gray.shape[:2]
    crop_h = int(h * ratio)
    crop_w = int(w * ratio)
    top = max(0, (h - crop_h) // 2)
    left = max(0, (w - crop_w) // 2)
    return gray[top : top + crop_h, left : left + crop_w]


def _extract_face_features(gray: np.ndarray) -> np.ndarray:
    import cv2

    crop = _center_crop(gray)
    crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    h, w = crop.shape
    left = crop[:, : w // 2]
    right = crop[:, w // 2 :]
    upper = crop[: h // 2, :]
    lower = crop[h // 2 :, :]
    center = crop[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    mouth = crop[2 * h // 3 :, w // 4 : 3 * w // 4]
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    mirrored = np.flip(right, axis=1)
    symmetry = float(np.mean(np.abs(left[:, : mirrored.shape[1]] - mirrored)))
    features = np.asarray(
        [
            float(crop.mean()),
            float(crop.std()),
            float(np.mean(np.abs(left.mean() - right.mean()))),
            float(np.mean(np.abs(upper.mean() - lower.mean()))),
            float(np.mean(np.abs(gx))),
            float(np.mean(np.abs(gy))),
            float(center.mean()),
            float(mouth.std()),
            float(symmetry),
            float(np.mean(np.abs(crop - crop.mean()))),
        ],
        dtype=np.float32,
    )
    return _clamp01(features / np.asarray([1, 0.5, 0.5, 0.5, 0.6, 0.6, 1, 0.5, 0.5, 0.6], dtype=np.float32))


def _extract_motion_features(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    import cv2

    prev = cv2.resize(prev_gray, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    curr = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    diff = np.abs(curr - prev)
    h, w = diff.shape
    upper = diff[: h // 2, :]
    lower = diff[h // 2 :, :]
    left = diff[:, : w // 2]
    right = diff[:, w // 2 :]
    ys, xs = np.mgrid[:h, :w]
    mass = diff.sum() + 1e-6
    cx = float((xs * diff).sum() / mass / w)
    cy = float((ys * diff).sum() / mass / h)
    features = np.asarray(
        [
            float(diff.mean()),
            float(diff.std()),
            float(upper.mean()),
            float(lower.mean()),
            float(left.mean()),
            float(right.mean()),
            float(cx),
            float(cy),
            float((diff > 0.08).mean()),
            float(np.max(diff)),
        ],
        dtype=np.float32,
    )
    return _clamp01(features / np.asarray([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1], dtype=np.float32))


def _extract_audio_sequence(wav_path: Path, steps: int) -> np.ndarray:
    sample_rate, waveform = wavfile.read(str(wav_path))
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)
    max_abs = np.max(np.abs(waveform)) + 1e-6
    waveform = waveform / max_abs
    indices = np.linspace(0, len(waveform), num=steps + 1).astype(np.int64)
    prev_rms = 0.0
    rows: List[np.ndarray] = []
    for idx in range(steps):
        start = indices[idx]
        end = max(indices[idx + 1], start + 1)
        chunk = waveform[start:end]
        freqs, _, spec = spectrogram(chunk, fs=sample_rate, nperseg=min(256, len(chunk)), noverlap=min(128, max(0, len(chunk) // 2)))
        power = np.mean(spec, axis=1) if spec.size else np.zeros((1,), dtype=np.float32)
        power_sum = float(power.sum()) + 1e-6
        centroid = float((freqs[: len(power)] * power).sum() / power_sum) if len(power) == len(freqs) else 0.0
        cumulative = np.cumsum(power)
        rolloff_idx = int(np.searchsorted(cumulative, 0.85 * power_sum))
        rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)]) if len(freqs) else 0.0
        low = float(power[freqs[: len(power)] < 400].sum() / power_sum) if len(freqs) else 0.0
        mid = float(power[(freqs[: len(power)] >= 400) & (freqs[: len(power)] < 2000)].sum() / power_sum) if len(freqs) else 0.0
        high = float(power[freqs[: len(power)] >= 2000].sum() / power_sum) if len(freqs) else 0.0
        rms = float(np.sqrt(np.mean(chunk**2) + 1e-8))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(chunk).astype(np.float32)))))
        bandwidth = float(np.sqrt(np.sum((((freqs[: len(power)] - centroid) ** 2) * power)) / power_sum)) if len(freqs) else 0.0
        rows.append(
            _clamp01(
                np.asarray(
                    [
                        rms,
                        zcr,
                        centroid / 4000.0,
                        bandwidth / 3000.0,
                        low,
                        mid,
                        high,
                        rolloff / 4000.0,
                        abs(rms - prev_rms) * 4.0,
                        float(rms < 0.03),
                    ],
                    dtype=np.float32,
                )
            )
        )
        prev_rms = rms
    return np.stack(rows, axis=0).astype(np.float32)


def _actor_split(actor_ids: Sequence[str]) -> Dict[str, set[str]]:
    unique = sorted(set(actor_ids))
    n = len(unique)
    train_cut = int(n * 0.7)
    dev_cut = int(n * 0.85)
    return {
        "train": set(unique[:train_cut]),
        "dev": set(unique[train_cut:dev_cut]),
        "test": set(unique[dev_cut:]),
    }


def _persona_placeholder(actor_id: str, persona_dim: int) -> np.ndarray:
    numeric = int(actor_id)
    base = np.asarray(
        [
            (numeric % 10) / 10.0,
            ((numeric // 10) % 10) / 10.0,
            ((numeric // 100) % 10) / 10.0,
            ((numeric // 1000) % 10) / 10.0,
            0.5,
            0.5,
        ],
        dtype=np.float32,
    )
    if persona_dim <= len(base):
        return base[:persona_dim]
    return np.pad(base, (0, persona_dim - len(base)), constant_values=0.5)


def build_cremad_trimodal_dataset(cfg: CREMADDatasetConfig) -> Dict[str, str]:
    repo_root = Path(cfg.repo_root)
    output_root = Path(cfg.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    video_root = repo_root / "VideoFlash"
    audio_root = repo_root / "AudioWAV"
    wav_files = sorted(audio_root.glob("*.wav"))
    actor_ids = [wav.stem.split("_")[0] for wav in wav_files]
    splits = _actor_split(actor_ids)
    buckets: Dict[str, Dict[str, List[np.ndarray]]] = {
        split: {"face": [], "motion": [], "audio": [], "persona": [], "emotion": [], "va": [], "state": [], "support_need": []}
        for split in ("train", "dev", "test")
    }
    manifest = {
        "source": "CREMA-D",
        "repo_root": str(repo_root),
        "sequence_steps": cfg.sequence_steps,
        "label_map": CREMAD_TO_FINE,
        "splits": {key: sorted(value) for key, value in splits.items()},
        "counts": {},
    }

    for wav_path in wav_files:
        stem = wav_path.stem
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        actor_id, _, emo = parts[:3]
        if emo not in CREMAD_TO_FINE:
            continue
        video_path = video_root / f"{stem}.flv"
        if not video_path.exists():
            continue
        frames = _iter_frames(video_path)
        if not frames:
            continue
        indices = _uniform_indices(len(frames), cfg.sequence_steps)
        sampled = [frames[int(i)] for i in indices]
        face = np.stack([_extract_face_features(frame) for frame in sampled], axis=0)
        motion_rows: List[np.ndarray] = []
        prev = sampled[0]
        for frame in sampled:
            motion_rows.append(_extract_motion_features(prev, frame))
            prev = frame
        motion = np.stack(motion_rows, axis=0)
        audio = _extract_audio_sequence(wav_path, cfg.sequence_steps)
        fine_label = CREMAD_TO_FINE[emo]
        emotion_idx = FINE_EMOTION_LABELS.index(fine_label)
        proto = EMOTION_PROTOTYPES[fine_label]
        va = np.asarray([proto["valence"], proto["arousal"]], dtype=np.float32)
        state = np.asarray([proto["stress"], proto["fatigue"], proto["attention_drop"], proto["suppression"]], dtype=np.float32)
        support_need = np.asarray(proto["stress"] * 0.4 + proto["fatigue"] * 0.2 + (1.0 - proto["valence"]) * 0.4, dtype=np.float32)
        split = "train" if actor_id in splits["train"] else "dev" if actor_id in splits["dev"] else "test"
        buckets[split]["face"].append(face)
        buckets[split]["motion"].append(motion)
        buckets[split]["audio"].append(audio)
        buckets[split]["persona"].append(_persona_placeholder(actor_id, cfg.persona_dim))
        buckets[split]["emotion"].append(np.asarray(emotion_idx, dtype=np.int64))
        buckets[split]["va"].append(va)
        buckets[split]["state"].append(state)
        buckets[split]["support_need"].append(support_need)

    outputs: Dict[str, str] = {}
    for split, payload in buckets.items():
        manifest["counts"][split] = int(len(payload["emotion"]))
        npz_payload = {key: np.stack(values, axis=0) for key, values in payload.items()}
        target = output_root / f"{split}.npz"
        np.savez_compressed(target, **npz_payload)
        outputs[split] = str(target)

    save_json(output_root / "manifest.json", manifest)
    return outputs
