# Paper Claim Guardrails

## Allowed Claims

- The repository implements an end-to-end research pipeline for proactive emotional care timing.
- The current system can generate synthetic data, train structured and multitask baselines, and evaluate them offline.
- The synthetic benchmark is internally consistent and useful for cold-start development.
- The structured baseline is a strong sanity check on the current synthetic rule family.
- The multitask model is an exploratory baseline for joint state and timing prediction.

## Disallowed Claims

- Do not say the model works on real users unless real-pilot data are actually evaluated.
- Do not call the current synthetic-only timing numbers evidence of real-world effectiveness.
- Do not claim clinical validity, therapeutic benefit, or psychological diagnosis.
- Do not claim the joint model is inferior in general; only say it underperformed on the current synthetic benchmark.
- Do not present accuracy as the main evidence when the class distribution is imbalanced.

## Required Wording

- Use `synthetic-only`, `simulated`, or `cold-start` whenever referring to current numbers.
- Use `exploratory` for the joint model.
- Use `sanity baseline` or `rule recovery` for the structured baseline.
- Use `feasibility` instead of `effectiveness` unless a real pilot supports stronger wording.

## Metrics Policy

- Timing results must emphasize `balanced_accuracy`, `macro_f1`, and per-class recall.
- `accuracy` can be reported, but it cannot be the headline metric.
- Any table with accuracy must also include class distribution or confusion matrix.

## Experimental Policy

- If a result is synthetic-only, label it that way in the table caption.
- If a result depends on hand-coded rules, say so directly.
- If a result uses a noiseless synthetic generator, call it a best-case sanity test, not a realistic benchmark.
- If a result is intended for the appendix, keep it out of the abstract and main claims.

## Example Phrasing

Allowed:

- "On our synthetic cold-start benchmark, the structured baseline recovers the generator's timing pattern well."
- "The multitask model is an exploratory joint model and currently underperforms the structured baseline on this benchmark."

Not allowed:

- "Our model can reliably decide when to intervene in real emotional care scenarios."
- "The proposed system outperforms prior work in proactive emotional care."
- "The joint model is worse than structured modeling in general."

