from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.io import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready experiment tables from metric JSON files.")
    parser.add_argument("--structured-synth", default=REPO_ROOT / "reports" / "structured_baseline_metrics.json")
    parser.add_argument("--structured-mixed", default=REPO_ROOT / "reports" / "structured_baseline_mixed_metrics.json")
    parser.add_argument("--multitask-synth", default=REPO_ROOT / "reports" / "multitask_metrics.json")
    parser.add_argument("--multitask-mixed", default=REPO_ROOT / "reports" / "multitask_mixed_metrics.json")
    parser.add_argument("--joint-mixed", default=REPO_ROOT / "reports" / "joint_metrics.json")
    parser.add_argument("--latex-out", default=REPO_ROOT / "paper" / "tables" / "main_results.tex")
    parser.add_argument("--weaklabel-out", default=REPO_ROOT / "paper" / "tables" / "weaklabel_results.tex")
    parser.add_argument("--summary-out", default=REPO_ROOT / "reports" / "formal_experiment_summary.md")
    return parser.parse_args()


def _pct(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{100.0 * value:.2f}"


def _mae(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:.3f}"


def _safe_metrics(path: Path) -> Dict[str, Any] | None:
    return load_json(path) if path.exists() else None


def _row(
    model_name: str,
    train_data: str,
    combined_metrics: Dict[str, Any],
    synthetic_metrics: Dict[str, Any],
    weak_metrics: Dict[str, Any] | None = None,
) -> str:
    return " & ".join(
        [
            model_name,
            train_data,
            _pct(combined_metrics.get("timing_balanced_accuracy")),
            _pct(combined_metrics.get("timing_macro_f1")),
            _pct(synthetic_metrics.get("timing_balanced_accuracy")),
            _pct(synthetic_metrics.get("timing_macro_f1")),
            _pct(weak_metrics.get("timing_macro_f1") if weak_metrics else None),
            _pct(combined_metrics.get("strategy_macro_f1")),
            _mae(combined_metrics.get("state_mae")),
        ]
    ) + r" \\"


def main() -> int:
    args = parse_args()
    structured_synth = _safe_metrics(Path(args.structured_synth))
    structured_mixed = _safe_metrics(Path(args.structured_mixed))
    multitask_synth = _safe_metrics(Path(args.multitask_synth))
    multitask_mixed = _safe_metrics(Path(args.multitask_mixed))
    joint_mixed = _safe_metrics(Path(args.joint_mixed))

    rows: List[str] = []
    if structured_synth:
        rows.append(_row("Structured", "Synthetic", structured_synth["metrics"]["test"], structured_synth["metrics"]["synthetic_test"]))
    if structured_mixed:
        rows.append(
            _row(
                "Structured",
                "Synthetic + weaklabel",
                structured_mixed["metrics"]["test"],
                structured_mixed["metrics"]["synthetic_test"],
                structured_mixed["metrics"].get("weaklabel_test"),
            )
        )
    if multitask_synth:
        rows.append(_row("Multitask temporal", "Synthetic", multitask_synth["metrics"]["test"], multitask_synth["metrics"]["synthetic_test"]))
    if multitask_mixed:
        rows.append(
            _row(
                "Multitask temporal",
                "Synthetic + weaklabel",
                multitask_mixed["metrics"]["test"],
                multitask_mixed["metrics"]["synthetic_test"],
                multitask_mixed["metrics"].get("weaklabel_test"),
            )
        )
    if joint_mixed:
        rows.append(
            _row(
                "Joint temporal",
                "Synthetic + weaklabel",
                joint_mixed["metrics"]["test"],
                joint_mixed["metrics"]["synthetic_test"],
                joint_mixed["metrics"].get("weaklabel_test"),
            )
        )

    latex = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Main offline results. Combined columns use the full held-out test split produced by each training setting; because that split contains 2{,}400 synthetic windows but only 62 weaklabel windows, the combined columns are synthetic-dominated. Synthetic columns evaluate only the held-out synthetic split, and the weaklabel column reports timing Macro-F1 on the held-out weaklabel split when available. Strategy scores remain template-like and should be interpreted as constrained planning sanity checks rather than open-ended response quality. All numbers are percentages except state MAE.}",
            r"\label{tab:main-results}",
            r"\small",
            r"\begin{tabular}{llccccccc}",
            r"\toprule",
            r"Model & Train Data & Comb. Bal. Acc. & Comb. Macro-F1 & Synth Bal. Acc. & Synth Macro-F1 & Weaklabel Macro-F1 & Comb. Strategy Macro-F1 & State MAE \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    Path(args.latex_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.latex_out).write_text(latex, encoding="utf-8")

    weak_rows: List[str] = []
    for title, payload in [
        ("Structured", structured_mixed),
        ("Multitask temporal", multitask_mixed),
        ("Joint temporal", joint_mixed),
    ]:
        if not payload or "weaklabel_test" not in payload["metrics"]:
            continue
        metrics = payload["metrics"]["weaklabel_test"]
        weak_rows.append(
            " & ".join(
                [
                    title,
                    _pct(metrics.get("timing_balanced_accuracy")),
                    _pct(metrics.get("timing_macro_f1")),
                    _pct(metrics.get("strategy_macro_f1")),
                ]
            )
            + r" \\"
        )
    weak_table = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Held-out weaklabel results on the 62-window database-derived split extracted from historical logs.}",
            r"\label{tab:weaklabel-results}",
            r"\small",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Model & Bal. Acc. & Macro-F1 & Strategy Macro-F1 \\",
            r"\midrule",
            *weak_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    Path(args.weaklabel_out).write_text(weak_table, encoding="utf-8")

    summary_lines = [
        "# Formal Experiment Summary",
        "",
        "This summary was auto-generated from the current metric JSON files.",
        "",
    ]
    for label, payload in [
        ("structured_synth", structured_synth),
        ("structured_mixed", structured_mixed),
        ("multitask_synth", multitask_synth),
        ("multitask_mixed", multitask_mixed),
        ("joint_mixed", joint_mixed),
    ]:
        if not payload:
            continue
        summary_lines.append(f"## {label}")
        summary_lines.append("")
        for split_name, split_metrics in payload["metrics"].items():
            summary_lines.append(f"- `{split_name}`: {split_metrics}")
        summary_lines.append("")
    Path(args.summary_out).write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"wrote {args.latex_out}")
    print(f"wrote {args.weaklabel_out}")
    print(f"wrote {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
