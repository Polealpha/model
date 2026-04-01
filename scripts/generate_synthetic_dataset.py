from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.config import load_config
from training.synthetic import write_synthetic_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic EpisodeWindowV1 datasets.")
    parser.add_argument("--config", default=REPO_ROOT / "training" / "configs" / "default.yaml")
    parser.add_argument("--output-root", default=REPO_ROOT / "datasets" / "synthetic")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    output_root = Path(args.output_root)
    if output_root.exists() and any(output_root.glob("*.jsonl")) and not args.force:
        print(f"synthetic dataset already exists at {output_root}")
        return 0
    synthetic_cfg = dict(cfg.get("synthetic", {}))
    counts = {
        "train": int(synthetic_cfg.get("train_size", 1200)),
        "dev": int(synthetic_cfg.get("dev_size", 240)),
        "test": int(synthetic_cfg.get("test_size", 240)),
    }
    write_synthetic_dataset(output_root, synthetic_cfg | {"seed": int(cfg.get("random_seed", 42))}, counts)
    print(f"wrote synthetic dataset to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

