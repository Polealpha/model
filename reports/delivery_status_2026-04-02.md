# Delivery Status 2026-04-02

## Current State

This repository now has a complete stage-1 research prototype for proactive emotional care timing:

- synthetic benchmark generation
- weaklabel extraction from `auth.db`
- structured baseline
- temporal multitask model
- temporal joint timing-to-strategy model
- dual-A800 training run
- CVPR-style paper source
- compiled PDF from the remote Linux server

## Final Offline Results

Source tables:

- `paper/tables/main_results.tex`
- `paper/tables/weaklabel_results.tex`

Key numbers:

- Structured / Synthetic:
  - balanced accuracy: `85.56`
  - timing Macro-F1: `79.11`
- Structured / Synthetic + weaklabel:
  - combined balanced accuracy: `83.81`
  - combined timing Macro-F1: `77.78`
  - weaklabel timing Macro-F1: `27.29`
- Multitask temporal / Synthetic:
  - balanced accuracy: `86.88`
  - timing Macro-F1: `77.21`
  - state MAE: `0.069`
- Multitask temporal / Synthetic + weaklabel:
  - combined balanced accuracy: `82.24`
  - combined timing Macro-F1: `76.15`
  - weaklabel timing Macro-F1: `7.62`
  - state MAE: `0.083`
- Joint temporal / Synthetic + weaklabel:
  - combined balanced accuracy: `85.79`
  - combined timing Macro-F1: `80.45`
  - synthetic timing Macro-F1: `81.06`
  - weaklabel timing Macro-F1: `36.29`
  - combined strategy Macro-F1: `82.66`
  - weaklabel strategy Macro-F1: `18.06`

## Paper Outputs

- Local compiled PDF:
  - `paper/main_remote_compiled.pdf`
- Remote compiled PDF:
  - `/root/lunwen_paper.pdf`
- Remote paper source:
  - `/root/lunwen_run/lunwen/paper`

## Remote Model Artifacts

Remote directory:

- `/root/lunwen_run/lunwen/training/artifacts`

Important files:

- `structured_baseline.pkl`
- `structured_baseline_mixed.pkl`
- `multitask_model.pt`
- `multitask_mixed_model.pt`
- `joint_model.pt`

## Remote Checksums

- `structured_baseline.pkl`: `e822980d9acea39f19a69109c8ee1f8de753dc5ebc25e5d4e47c13f7ffbd01bb`
- `structured_baseline_mixed.pkl`: `249bc1cbf84e9fa843210767267bf6894a1956894ccca7289274852a2d62e24f`
- `multitask_model.pt`: `8062162782a215e11aefeb8acd33cbf13f48ac5eb39104069dbfce605eaae2d0`
- `multitask_mixed_model.pt`: `7ca0bd4097ef94f723bd6793e0e93e08c2dab2af84ffa6d378fa3c0ef4e006a8`
- `joint_model.pt`: `056b3ac9ade4351b84f8191ddda82cc988e0e9a5521487604cc8ad93d01db2c7`

## What Is Strong Enough To Claim

- the task decomposition is implemented end-to-end
- the repository supports synthetic plus weaklabel bootstrapping
- the temporal descriptor backbone trains successfully on dual A800 GPUs
- the joint branch is the strongest current offline timing model
- weaklabel generalization remains the main bottleneck

## What Is Not Yet Strong Enough To Claim

- raw-video end-to-end performance
- validated personality benefit through ablation
- robust real-world timing performance
- strong support-generation quality from real user outcomes
- clinical or deployment efficacy

## Highest-Impact Next Steps

1. Add raw video/audio ingestion so the backbone is no longer descriptor-only.
2. Expand weaklabel extraction with `CarePlanReady`, `CarePlanSkipped`, and bridge logs.
3. Run personality ablations and calibration analysis.
4. Start the real pilot with acceptance / annoyance / response-latency logging.
5. Prepare the repo for GitHub publication and reproducibility packaging.
