# Research Engine

This package implements the clean-room research engine for proactive emotional care.

## Subpackages

- `schemas`: frozen interfaces shared by data, models, evaluation, and paper tables
- `state`: affective state estimation helpers
- `policy`: rule-based timing and strategy decisions
- `dialogue`: structured strategy-to-utterance pipeline
- `simulator`: synthetic data generation
- `evaluation`: metrics and replay-oriented summaries

## Design rules

1. Keep timing prediction separate from utterance generation.
2. Use structured interfaces between stages.
3. Treat false interruption as a primary failure mode.
4. Do not import the dirty `emotion engine` repo directly.

