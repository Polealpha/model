# Training

This package contains the research-side implementation for proactive emotional care timing.

## Layout

- `configs/`: default hyperparameters and path conventions
- `baselines/`: structured baseline models
- `multitask/`: shared representation model for state + timing
- `joint/`: shared representation model for timing + strategy + template
- `schema.py`: frozen EpisodeWindowV1 and output dataclasses
- `synthetic.py`: synthetic data generation and dataset writer
- `feature_utils.py`: feature flattening and target extraction
- `simple_mlp.py`: minimal shared MLP used by multitask and joint models

## Default data flow

1. Generate synthetic splits under `datasets/synthetic`
2. Train the structured baseline on `train.jsonl`
3. Train the multitask model on the same split
4. Train the joint model if needed for strategy/template studies
5. Save model artifacts under `training/artifacts`

## Expected artifacts

- Structured baseline: pickled sklearn pipelines
- Multitask/joint models: pickled shared MLP artifacts plus metadata

