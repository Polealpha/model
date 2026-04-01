from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


@dataclass
class HeadSpec:
    kind: str
    dim: int


class SharedMultiHeadMLP:
    def __init__(
        self,
        input_dim: int,
        head_specs: Mapping[str, HeadSpec],
        hidden_dim: int = 64,
        lr: float = 0.01,
        epochs: int = 50,
        l2: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.head_specs = dict(head_specs)
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.rng = np.random.default_rng(seed)
        self.scaler = StandardScaler()
        self.W1 = self.rng.normal(scale=0.1, size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.head_W: Dict[str, np.ndarray] = {}
        self.head_b: Dict[str, np.ndarray] = {}
        for name, spec in self.head_specs.items():
            self.head_W[name] = self.rng.normal(scale=0.1, size=(hidden_dim, spec.dim))
            self.head_b[name] = np.zeros(spec.dim, dtype=np.float32)

    def fit(self, X: np.ndarray, targets: Mapping[str, np.ndarray]) -> "SharedMultiHeadMLP":
        Xs = self.scaler.fit_transform(X)
        n = Xs.shape[0]
        if n == 0:
            raise ValueError("Cannot fit on empty dataset")
        for _ in range(self.epochs):
            h_pre = Xs @ self.W1 + self.b1
            h = np.tanh(h_pre)
            grads_h = np.zeros_like(h)
            head_grads: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

            for name, spec in self.head_specs.items():
                W = self.head_W[name]
                b = self.head_b[name]
                y = np.asarray(targets[name])
                if spec.kind == "categorical":
                    y_idx = y.astype(int)
                    y_oh = np.eye(spec.dim, dtype=np.float32)[y_idx]
                    logits = h @ W + b
                    probs = softmax(logits)
                    grad_logits = (probs - y_oh) / n
                    loss_grad = h.T @ grad_logits + self.l2 * W
                    bias_grad = grad_logits.sum(axis=0)
                    grads_h += grad_logits @ W.T
                    head_grads[name] = (loss_grad, bias_grad)
                elif spec.kind == "regression":
                    y = y.astype(np.float32)
                    pred = h @ W + b
                    grad_pred = 2.0 * (pred - y) / n
                    loss_grad = h.T @ grad_pred + self.l2 * W
                    bias_grad = grad_pred.sum(axis=0)
                    grads_h += grad_pred @ W.T
                    head_grads[name] = (loss_grad, bias_grad)
                else:
                    raise ValueError(f"Unsupported head kind: {spec.kind}")

            tanh_grad = (1.0 - h ** 2) * grads_h
            grad_W1 = Xs.T @ tanh_grad + self.l2 * self.W1
            grad_b1 = tanh_grad.sum(axis=0)

            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1
            for name, (grad_W, grad_b) in head_grads.items():
                self.head_W[name] -= self.lr * grad_W
                self.head_b[name] -= self.lr * grad_b
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return np.tanh(Xs @ self.W1 + self.b1)

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        h = self.transform(X)
        outputs: Dict[str, np.ndarray] = {}
        for name, spec in self.head_specs.items():
            logits = h @ self.head_W[name] + self.head_b[name]
            if spec.kind == "categorical":
                outputs[name] = softmax(logits)
            else:
                outputs[name] = logits
        return outputs

    def save(self, path: str | Path, metadata: Dict | None = None) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_dim": self.input_dim,
            "head_specs": self.head_specs,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "epochs": self.epochs,
            "l2": self.l2,
            "scaler": self.scaler,
            "W1": self.W1,
            "b1": self.b1,
            "head_W": self.head_W,
            "head_b": self.head_b,
            "metadata": metadata or {},
        }
        with target.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load(cls, path: str | Path) -> "SharedMultiHeadMLP":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        model = cls(
            input_dim=payload["input_dim"],
            head_specs=payload["head_specs"],
            hidden_dim=payload["hidden_dim"],
            lr=payload["lr"],
            epochs=payload["epochs"],
            l2=payload["l2"],
        )
        model.scaler = payload["scaler"]
        model.W1 = payload["W1"]
        model.b1 = payload["b1"]
        model.head_W = payload["head_W"]
        model.head_b = payload["head_b"]
        return model

