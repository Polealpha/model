# Paper Claim Review Round 2

Overall judgment: the draft is mostly inside the current guardrails, but a few claims still read stronger than the evidence shown in the paper tables. The biggest risk is not blatant fabrication; it is that readers can over-interpret mixed synthetic+weaklabel headline numbers as broader robustness than the current evidence supports.

## Highest-Risk Issues

1. `abstract.tex`: "three offline data sources" is inaccurate.
The text counts a "pending real-pilot protocol" as an offline data source. That is a protocol, not a released data source.
Suggested fix: change this to "two offline data sources ... plus a pending real-pilot protocol."

2. `abstract.tex`, `introduction.tex`, `experiments.tex`, `results_discussion.tex`: the "combined synthetic+weaklabel test set" headline numbers are not directly supported by the visible main tables.
The prose emphasizes `84.09 / 79.45`, `81.75 / 76.14`, and `71.93 / 71.89`, but [main_results.tex] only shows synthetic-test metrics plus a weaklabel-only Macro-F1 column. A reader cannot verify the combined-set headline from the presented tables alone.
Suggested fix: either add an explicit combined-set table/footnote, or rewrite the prose so the headline numbers are the ones actually shown in the tables.

3. `experiments.tex`, `results_discussion.tex`, `limitations_ethics.tex`, `main_results.tex`: the paper does not say clearly enough that the synthetic timing benchmark nearly excludes the `immediate` class.
`formal_experiment_summary.md` shows synthetic test support for `immediate` is `1`, and all synthetic-only models have `immediate` F1 = `0.0`. This is a major caveat for interpreting synthetic balanced accuracy and macro-F1.
Suggested fix: explicitly state that the synthetic held-out split is a rule-recovery sanity test with severe class skew, and do not let synthetic timing numbers stand alone without that caveat.

## Medium-Risk Wording Problems

4. `introduction.tex`: "strong structured model" is stronger than the repo's own claim policy.
The guardrail prefers `sanity baseline` or `rule recovery` for the structured branch.
Suggested fix: replace "strong structured model" with "structured sanity baseline" or "rule-recovery baseline."

5. `results_discussion.tex`: "the new GPU-backed temporal path is viable" is not directly evidenced in the reviewed paper/result files.
The reviewed materials show offline metrics, not hardware scaling or utilization evidence.
Suggested fix: say "the temporal backbone trains end-to-end in the current repository and improves over the earlier shared-MLP prototype" or remove the hardware implication.

6. `main_results.tex`: the caption still underspecifies how synthetic the strategy scores are.
The `94.02` synthetic strategy Macro-F1 for the joint model is easy to over-read, but the discussion already admits those labels are template-like.
Suggested fix: add a caption note that synthetic strategy scores are best-case sanity numbers on template-like labels, not evidence of open-ended supportive generation quality.

## Caveats That Should Be Pulled Forward

7. `abstract.tex` and `experiments.tex`: weaklabel-only generalization should appear earlier in the story.
Right now the main narrative foregrounds the mixed headline numbers, while the held-out weaklabel timing Macro-F1 remains only `42.45`, `39.65`, and `39.94`.
Suggested fix: add one short clause in the abstract or main-results paragraph saying that weaklabel-only performance remains below `43\%` Macro-F1, so the reader does not confuse mixed-set gains with real-world robustness.

## What Already Looks Correct

- `limitations_ethics.tex` is directionally aligned with the guardrails: it avoids clinical claims and explicitly says the system is not end-to-end raw video/audio in the wild.
- `results_discussion.tex` already handles the joint model more carefully than before by calling it exploratory and by warning that synthetic strategy performance is inflated.
- The draft does not misuse plain accuracy as the headline metric; balanced accuracy and macro-F1 remain central.
