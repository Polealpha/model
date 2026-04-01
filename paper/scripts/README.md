# Scripts

This folder is reserved for the scripts that will eventually populate the paper figures, tables, and experiment logs.

Expected future utilities:

- `generate_synthetic_dataset.py`: build `synthetic/` windows and labels.
- `build_weak_labels.py`: derive partial supervision from existing logs.
- `train_rule_baseline.py`: reproduce the deterministic timing policy.
- `train_structured_baseline.py`: train feature-based timing models.
- `train_multitask_model.py`: train the video-persona multitask network.
- `train_joint_model.py`: train the timing-to-strategy branch.
- `evaluate.py`: compute offline metrics and export tables.
- `export_figures.py`: render paper-ready plots into `figures/`.
- `export_tables.py`: render paper-ready tables into `tables/`.

When these scripts are added, they should write outputs only under `paper/reports`, `paper/figures`, and `paper/tables` so the manuscript remains self-contained.

