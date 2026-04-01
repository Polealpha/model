from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.weaklabel import WeakLabelConfig, write_weaklabel_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract weak-label episodes from the legacy auth.db logs.")
    parser.add_argument("--db", default=Path(r"E:\Desktop\emotion engine\backend\auth.db"))
    parser.add_argument("--output-root", default=REPO_ROOT / "datasets" / "weaklabel")
    parser.add_argument("--include-bridge", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = write_weaklabel_dataset(args.db, args.output_root, WeakLabelConfig(include_bridge=args.include_bridge))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
