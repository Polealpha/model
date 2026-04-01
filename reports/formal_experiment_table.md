# Formal Experiment Table

| Setting | Model | Timing Acc | Timing Bal Acc | Timing Macro-F1 | Strategy Acc | Strategy Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| synthetic_only | structured | 0.9125 | 0.6164 | 0.6037 | 0.7333 | 0.4481 |
| synthetic_only | multitask | 0.8750 | 0.5868 | 0.5761 | nan | nan |
| synthetic_only | joint | 0.8375 | 0.5307 | 0.5381 | 0.8500 | 0.8556 |
| synthetic_plus_weaklabel_on_synth_test | structured | 0.9125 | 0.6164 | 0.6037 | 0.9167 | 0.9270 |
| synthetic_plus_weaklabel_on_synth_test | multitask | 0.8792 | 0.5911 | 0.5794 | nan | nan |
| synthetic_plus_weaklabel_on_synth_test | joint | 0.8542 | 0.5500 | 0.5540 | 0.8667 | 0.8769 |
| synthetic_plus_weaklabel_on_weaklabel_test | structured | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| synthetic_plus_weaklabel_on_weaklabel_test | multitask | 1.0000 | 1.0000 | 1.0000 | nan | nan |
| synthetic_plus_weaklabel_on_weaklabel_test | joint | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
