from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.config import load_config
from training.trimodal_emotion import write_trimodal_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic tri-modal fine-grained emotion data.")
    parser.add_argument("--config", default=REPO_ROOT / "training" / "configs" / "remote_a800_trimodal.yaml")
    parser.add_argument("--root", default=REPO_ROOT / "datasets" / "trimodal_emotion")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if root.exists() and not args.force and (root / "manifest.json").exists():
        print(f"dataset already exists at {root}")
        return 0
    cfg = load_config(args.config)
    data_cfg = dict(cfg.get("trimodal_synthetic", {}))
    counts = {
        "train": int(data_cfg.get("train_size", 120000)),
        "dev": int(data_cfg.get("dev_size", 12000)),
        "test": int(data_cfg.get("test_size", 12000)),
    }
    write_trimodal_dataset(root, data_cfg, counts)
    print(f"wrote trimodal dataset to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
