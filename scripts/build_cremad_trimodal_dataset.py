from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.public_cremad import CREMADDatasetConfig, build_cremad_trimodal_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a tri-modal dataset from CREMA-D.")
    parser.add_argument("--repo-root", default="/root/datasets_public/crema-d-mirror")
    parser.add_argument("--output-root", default=REPO_ROOT / "datasets" / "public_cremad_trimodal")
    parser.add_argument("--sequence-steps", type=int, default=48)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = CREMADDatasetConfig(
        repo_root=str(args.repo_root),
        output_root=str(args.output_root),
        sequence_steps=int(args.sequence_steps),
    )
    outputs = build_cremad_trimodal_dataset(cfg)
    print(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
