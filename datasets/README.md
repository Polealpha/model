# Datasets

This repository uses three data layers and two derived layers.

- `synthetic`: generated windows used for cold-start training and ablations
- `weaklabel`: weak supervision exported from historical rule triggers and confirmation flows
- `realpilot`: small-scale real-user pilot samples with explicit feedback
- `processed`: cached feature tables and model-ready arrays
- `splits`: train/val/test split manifests

## Storage policy

- Do not store long raw video or audio by default.
- Prefer short windows and structured features.
- Keep synthetic artifacts reproducible from scripts when possible.

