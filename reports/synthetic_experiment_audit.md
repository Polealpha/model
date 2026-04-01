# Synthetic Experiment Audit

## Scope

This audit reviews the current synthetic-only pipeline:

- synthetic generation in `training/synthetic.py`
- feature extraction in `training/feature_utils.py`
- structured baseline in `training/baselines/structured.py`
- multitask model in `training/multitask/model.py`

Reported test numbers at audit time:

- structured: `timing_accuracy=0.9125`, `timing_macro_f1=0.6037`
- multitask: `timing_accuracy=0.8042`, `timing_macro_f1=0.5321`

These results are synthetic-only and must not be presented as real-user performance.

## Findings

### 1. No direct split leakage, but the split protocol is still weak

The current dataset is generated with a single RNG stream and then written into train/dev/test files in sequence. That does not create literal sample overlap, and the feature set does not include split markers or episode IDs. However, the split is not stratified by timing or strategy labels, so the dev/test distributions can drift sharply from train. In the generated data, `immediate` is almost absent in dev/test, which makes evaluation unstable.

Action:

- regenerate splits with stratification over `timing_label` and `strategy_label`
- for future real data, split by subject/persona/time, not by window

### 2. The synthetic labels are too close to the rule baseline

The generator uses hand-coded latent rules to create both `timing_label` and `strategy_label`. The structured baseline then learns from features that were themselves engineered from the same latent factors. This is useful for a cold start, but it means the structured baseline is mostly recovering the synthetic rule, not discovering a general intervention policy.

Action:

- treat structured baseline as a sanity baseline, not a research claim
- add at least one noisier generator mode with label noise, missingness, and conflicting cues
- report a majority-class baseline and a simple linear baseline alongside the structured model

### 3. Accuracy is misleading here because the classes are heavily imbalanced

On the current synthetic splits, `timing_label` is dominated by `none` and `delay`, with almost no `immediate` samples in dev/test. In that setting, raw accuracy can hide poor minority-class recall. The reported `macro_f1` is better, but it is still unstable because the rare class has too few examples.

Action:

- always report per-class precision/recall/F1
- keep `balanced_accuracy` and `macro_f1` as the primary timing metrics
- add confusion matrices and minority-class recall for `immediate`
- do not present accuracy as the headline metric

### 4. The multitask model is structurally disadvantaged on this dataset

The multitask model is a small tanh MLP trained jointly for state regression and timing classification. On a tiny synthetic tabular dataset, a convex structured baseline will often outperform a jointly trained neural model because:

- the baseline matches the data geometry better
- the multitask model has limited capacity
- the regression head can dominate shared gradients
- there is no explicit loss reweighting or calibration

This makes the current multitask result useful as an exploratory baseline, not as evidence that joint modeling is inferior in principle.

Action:

- if joint training remains important, add explicit loss weights and early stopping
- compare against a timing-only MLP, not just the multitask model
- use the multitask model as an ablation, not the main result

### 5. The current numbers cannot support strong paper claims

These claims are not supported yet:

- the model generalizes to real-world proactive care
- the timing policy is robust across people or scenes
- the joint model improves over a strong structured baseline
- the system can infer the best care moment from video and persona in the wild

What the current results do support is narrower:

- the synthetic generator is internally consistent
- the structured baseline can recover the synthetic rule reasonably well
- the training and evaluation pipeline is executable end to end

## Recommended Reporting

For the paper, the current synthetic experiment should be framed as:

- a cold-start feasibility study
- a benchmark sanity check
- a proof that the pipeline can train and evaluate

It should not be framed as:

- evidence of real-world effectiveness
- evidence of clinical relevance
- evidence that the joint model is broadly worse than structured modeling

## Next Steps

1. Regenerate synthetic data with stratified splits and more minority-class examples.
2. Add a majority-class baseline and a timing-only MLP baseline.
3. Add per-class metrics and confusion matrices to the report generator.
4. Introduce a noisier synthetic mode before claiming any model comparison.
5. Keep the current synthetic-only results in the appendix or implementation notes, not the abstract.

