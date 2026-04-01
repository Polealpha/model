from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile the CVPR paper if the LaTeX source exists.")
    parser.add_argument("--paper-dir", default=REPO_ROOT / "paper")
    parser.add_argument("--main-tex", default="main.tex")
    parser.add_argument("--output-dir", default=REPO_ROOT / "paper" / "build")
    parser.add_argument("--reports-dir", default=REPO_ROOT / "reports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paper_dir = Path(args.paper_dir)
    main_tex = paper_dir / args.main_tex
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.reports_dir) / "paper_compile_status.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if not main_tex.exists():
        payload = {"status": "missing_main_tex", "main_tex": str(main_tex)}
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    compiler = None
    for candidate in ("latexmk", "pdflatex", "tectonic"):
        if shutil.which(candidate):
            compiler = candidate
            break

    if compiler is None:
        payload = {"status": "no_latex_compiler", "main_tex": str(main_tex)}
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 3

    if compiler == "latexmk":
        cmd = [compiler, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-outdir=" + str(output_dir), str(main_tex)]
    elif compiler == "pdflatex":
        cmd = [compiler, "-interaction=nonstopmode", "-halt-on-error", "-output-directory", str(output_dir), str(main_tex)]
    else:
        cmd = [compiler, "--outdir", str(output_dir), str(main_tex)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    payload = {
        "status": "compiled" if result.returncode == 0 else "compile_failed",
        "compiler": compiler,
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "main_tex": str(main_tex),
        "output_dir": str(output_dir),
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

