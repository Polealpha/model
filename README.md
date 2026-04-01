# Active Proactive Care Research Repo

This repository hosts the first research-grade implementation for proactive emotional care timing, strategy selection, and CVPR-style paper writing.

## Repository goals

- Use the latest Windows desktop client as the product-facing UI base.
- Build a clean research engine around `video/persona/context -> state -> timing -> strategy`.
- Train and evaluate initial baselines on synthetic data first, then calibrate with weak labels and small real pilots.
- Keep the paper system, experiments, and data contracts in the same repo.

## Layout

- `app/desktop`: desktop client source migrated from the latest Windows app.
- `engine`: research engine, contracts, heuristics, simulator, and evaluation helpers.
- `datasets`: synthetic, weak-label, real-pilot, processed, and split artifacts.
- `training`: baseline and model training code.
- `scripts`: data generation, training, evaluation, and paper utilities.
- `paper`: CVPR-style paper source and bibliography.
- `reports`: experiment logs and result templates.
- `docs`: labeling, ethics, protocol, and reproducibility notes.

## Implementation status

- Repo bootstrap: in progress
- Desktop app migration: source-only migration completed
- Research engine: rule baseline and interfaces implemented
- Synthetic data pipeline: implemented
- Structured baseline: implemented through `scripts/train_structured_baseline.py`
- Multitask and joint model: scaffolded for follow-up training
- Paper system: CVPR-style scaffold in progress

## First steps

1. Create a synthetic dataset.
2. Train the structured baseline.
3. Evaluate offline timing prediction.
4. Fill paper tables from generated reports.

