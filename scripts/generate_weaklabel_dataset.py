from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.weaklabel import write_weaklabel_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weaklabel datasets from emotion engine logs.")
    parser.add_argument("--output-root", default=REPO_ROOT / "datasets" / "weaklabel")
    parser.add_argument(
        "--log-path",
        action="append",
        dest="log_paths",
        default=[],
        help="Log path containing CarePlanReady entries. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_paths = args.log_paths or [
        r"E:\Desktop\emotion engine\logs\bridge_latest.log",
        r"E:\Desktop\emotion engine\runtime_logs\bridge.log",
        r"E:\Desktop\emotion engine\runtime_logs\bridge_live.log",
    ]
    outputs = write_weaklabel_dataset(args.output_root, log_paths)
    print(outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

