from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.config import load_config
from training.data import load_episodes
from training.io import save_json
from training.baselines.structured import StructuredBaseline
from training.multitask.model import MultiTaskModel
from training.joint.model import JointModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build formal experiment summaries for synthetic and synthetic+weaklabel settings.")
    parser.add_argument("--config", default=REPO_ROOT / "training" / "configs" / "default.yaml")
    parser.add_argument("--synthetic-root", default=REPO_ROOT / "datasets" / "synthetic")
    parser.add_argument("--weaklabel-root", default=REPO_ROOT / "datasets" / "weaklabel")
    parser.add_argument("--output-json", default=REPO_ROOT / "reports" / "formal_experiment_table.json")
    parser.add_argument("--output-md", default=REPO_ROOT / "reports" / "formal_experiment_table.md")
    return parser.parse_args()


def _timing_idx(decision: str) -> int:
    return 0 if decision == "none" else 1 if decision == "delay" else 2


def _strategy_idx(level: str) -> int:
    return 0 if level == "observe" else 1 if level == "nudge" else 2 if level == "care" else 3


def _eval(model, episodes):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    preds = model.predict(episodes)
    timing_gold = [_timing_idx(ep.timing_label) for ep in episodes]
    timing_pred = [_timing_idx(item["timing"]["decision"]) for item in preds]
    row = {
        "timing_accuracy": float(accuracy_score(timing_gold, timing_pred)),
        "timing_balanced_accuracy": float(balanced_accuracy_score(timing_gold, timing_pred)),
        "timing_macro_f1": float(f1_score(timing_gold, timing_pred, average="macro")),
    }
    if "strategy" in preds[0]:
        strategy_gold = [_strategy_idx(ep.strategy_label) for ep in episodes]
        strategy_pred = [_strategy_idx(item["strategy"]["strategy_level"]) for item in preds]
        row["strategy_accuracy"] = float(accuracy_score(strategy_gold, strategy_pred))
        row["strategy_macro_f1"] = float(f1_score(strategy_gold, strategy_pred, average="macro"))
    return row


def _train_and_eval(setting_name: str, train_eps, eval_eps, cfg):
    structured = StructuredBaseline(cfg.get("structured_baseline", {})).fit(train_eps)
    multitask = MultiTaskModel(cfg.get("multitask", {})).fit(train_eps)
    joint = JointModel(cfg.get("joint", {})).fit(train_eps)
    return {
        "setting": setting_name,
        "structured": _eval(structured, eval_eps),
        "multitask": _eval(multitask, eval_eps),
        "joint": _eval(joint, eval_eps),
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    synthetic_root = Path(args.synthetic_root)
    weaklabel_root = Path(args.weaklabel_root)

    synth_train = load_episodes(synthetic_root / "train.jsonl")
    synth_test = load_episodes(synthetic_root / "test.jsonl")
    weak_train = load_episodes(weaklabel_root / "train.jsonl") if (weaklabel_root / "train.jsonl").exists() else []
    weak_test = load_episodes(weaklabel_root / "test.jsonl") if (weaklabel_root / "test.jsonl").exists() else []

    rows = [
        _train_and_eval("synthetic_only", synth_train, synth_test, cfg),
    ]
    if weak_train:
        rows.append(_train_and_eval("synthetic_plus_weaklabel_on_synth_test", synth_train + weak_train, synth_test, cfg))
    if weak_train and weak_test:
        rows.append(_train_and_eval("synthetic_plus_weaklabel_on_weaklabel_test", synth_train + weak_train, weak_test, cfg))

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output_json, rows)
    lines = [
        "# Formal Experiment Table",
        "",
        "| Setting | Model | Timing Acc | Timing Bal Acc | Timing Macro-F1 | Strategy Acc | Strategy Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for model_name in ("structured", "multitask", "joint"):
            metrics = row[model_name]
            lines.append(
                f"| {row['setting']} | {model_name} | {metrics.get('timing_accuracy', float('nan')):.4f} | {metrics.get('timing_balanced_accuracy', float('nan')):.4f} | {metrics.get('timing_macro_f1', float('nan')):.4f} | {metrics.get('strategy_accuracy', float('nan')):.4f} | {metrics.get('strategy_macro_f1', float('nan')):.4f} |"
            )
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
