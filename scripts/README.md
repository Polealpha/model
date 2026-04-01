# Scripts

This folder contains the runnable entrypoints for the research workflow.

## Commands

- `python scripts/generate_synthetic_dataset.py`
- `python scripts/train_structured_baseline.py`
- `python scripts/train_multitask_model.py`
- `python scripts/evaluate_predictions.py --model-path ... --data-path ...`
- `python scripts/compile_paper.py`

## Conventions

- Default config: `training/configs/default.yaml`
- Default synthetic data root: `datasets/synthetic`
- Model artifacts are saved under `training/artifacts`
- Evaluation summaries are written to `reports`

