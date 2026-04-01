# Paper Scaffold

This directory contains the CVPR-style paper skeleton for
`Video-to-Intervention: Predicting Receptivity and Support Strategy for Proactive Emotional Care`.

## What is already written

- A full LaTeX paper skeleton in `main.tex`.
- Draft text for all core sections under `sections/`.
- A curated bibliography in `refs.bib`.
- Figure and table placeholder directories.
- A short guide for future automation scripts in `scripts/README.md`.

## Build

Default local build:

```bash
latexmk -pdf main.tex
```

Fallback build:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

If the official CVPR style file `cvpr.sty` is available in the same directory, `main.tex` will use it automatically. Otherwise it falls back to a standard two-column article layout so the skeleton remains buildable.

## Section map

- `sections/abstract.tex`: paper summary and scope
- `sections/introduction.tex`: problem statement, motivation, and contributions
- `sections/related_work.tex`: five-part related work framing
- `sections/task_definition.tex`: formal task and interface definitions
- `sections/dataset_construction.tex`: synthetic, weak-label, and pilot data protocol
- `sections/method.tex`: model architecture and loss design
- `sections/experiments.tex`: evaluation protocol, baselines, and ablations
- `sections/human_eval.tex`: user study and wizard-of-oz protocol
- `sections/limitations_ethics.tex`: privacy, safety, and medical boundary
- `sections/conclusion.tex`: closing summary and next steps

## What is still pending

- Real experiment results.
- Auto-filled tables and figures from training runs.
- Final camera-ready polishing against the official CVPR template if that template is added later.

