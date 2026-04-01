# Reproducibility

## Determinism

- Default random seed: `7`
- Synthetic data generation and structured baseline training must accept explicit seeds.

## Report artifacts

- Store metrics as JSON in `reports/generated/`
- Store experiment notes in Markdown beside metrics
- Keep paper tables derived from report artifacts rather than manually edited numbers

## Current limitations

- Remote A800 training is not wired in yet because the provided SSH credentials were rejected during read-only verification.
- Real pilot data collection is scaffolded, not yet populated.
- Joint generation remains exploratory in v1.

