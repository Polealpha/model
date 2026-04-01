# Paper-Ready Results Summary

## Scope

All numbers below are from the synthetic-only release on the generated train/dev/test split with 1200/240/240 episodes. No weak-label, real-pilot, or end-to-end video-backbone results are included yet.

## Main Results

| Split | Structured Timing Acc. | Structured Bal. Acc. | Structured Macro-F1 | Structured Strategy Macro-F1 | Multitask Timing Acc. | Multitask Bal. Acc. | Multitask Macro-F1 | Multitask State MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 89.83 | 90.70 | 89.17 | 44.16 | 77.83 | 79.65 | 51.91 | 0.118 |
| Dev | 90.83 | 60.98 | 59.97 | 44.19 | 76.67 | 53.08 | 50.94 | 0.122 |
| Test | 91.25 | 61.64 | 60.37 | 44.81 | 80.42 | 55.16 | 53.21 | 0.123 |

## Recommended Paper Wording

- The current results are synthetic-only and should be framed as a pipeline validation study.
- The structured baseline is stronger than the multitask model on timing metrics under the current synthetic release.
- Balanced accuracy and macro-F1 should be emphasized over accuracy because the timing labels remain imbalanced.
- The multitask model should be described as a proof of interface compatibility, not as the final best model.
- Claims about real-user benefit, clinical efficacy, or cross-scene robustness are not yet supported.

## Pending Dependencies

- `joint` timing-to-strategy training is still in progress.
- The video backbone has not yet been connected to the training loop.
- Weak-label and real-pilot experiments are not yet reported.

