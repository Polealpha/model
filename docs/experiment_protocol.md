# Experiment Protocol

## Offline experiments

1. Generate synthetic train/val/test splits.
2. Train the rule baseline and structured baseline.
3. Train the multitask model.
4. Train the exploratory joint model.
5. Export metrics and ablation summaries into `reports/`.

## Human or Wizard-of-Oz pilot

- Use the same `EpisodeWindowV1` contract.
- Collect explicit feedback: `accepted`, `ignored`, `annoyed`, and optional short self-report.
- Report acceptance and annoyance separately from classification metrics.

## Required ablations

- `video-only`
- `persona-only`
- `context-only`
- `video + persona`
- `video + persona + context`
- `pipeline vs joint`
- `with vs without state supervision`
- `synthetic only / synthetic + weaklabel / synthetic + weaklabel + realpilot`

