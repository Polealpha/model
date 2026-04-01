# Public Dataset And Qualification Plan 2026-04-02

## Why This Matters

Current synthetic tri-modal results are useful for validating architecture and modality fusion, but they are not enough to claim the model is good in real environments.

For this project, "qualified" must mean two things at the same time:

1. the upstream tri-modal perception model is strong on clean public affect datasets
2. the downstream proactive-care timing model is strong enough on weaklabel and later real-pilot data

If only the first is good, we merely have an emotion recognizer.
If only the second is good on synthetic data, we merely have a synthetic decision engine.
The project target requires both.

## Recommended Clean Public Datasets

### 1. Aff-Wild2

Primary use:

- face / video expression recognition
- valence-arousal regression
- in-the-wild affect robustness

Why use it:

- large audiovisual in-the-wild benchmark
- annotated for expression classification, valence-arousal, and AUs
- good first public pretraining source for the face branch

Primary source:

- official dataset page: [Aff-Wild2](https://sites.google.com/view/dimitrioskollias/databases/aff-wild2)

Important note:

- access requires EULA / academic request

### 2. MAFW

Primary use:

- video + audio multimodal fine-grained emotion pretraining
- compound emotion discrimination

Why use it:

- explicitly multi-modal
- in-the-wild
- 10,045 clips
- single-expression and multiple-expression settings

Primary source:

- official dataset page: [MAFW](https://mafw-database.github.io/MAFW/)

Important note:

- access requires EULA / faculty-backed request for students

### 3. IEMOCAP

Primary use:

- audiovisual + motion supervision
- categorical emotion and dimensional affect
- direct support for the "face + motion + audio" requirement

Why use it:

- contains video, speech, motion capture, transcriptions
- includes categorical labels and valence/activation/dominance-style labels
- best clean public source for the motion/action branch

Primary source:

- official dataset page: [IEMOCAP](https://sail.usc.edu/iemocap/index.html)

Important note:

- licensed access
- smaller than Aff-Wild2/MAFW, so better used as supervised finetuning or motion alignment source, not the only pretraining source

### 4. MELD

Primary use:

- conversational multimodal emotion
- downstream context-aware calibration

Why use it:

- audio + video + text
- emotion shifts in dialogue
- useful once we move from isolated windows toward interaction-aware care timing

Primary sources:

- official GitHub / data page: [MELD GitHub](https://github.com/declare-lab/MELD)
- project site: [MELD site](https://affective-meld.github.io/)

Important note:

- not a direct replacement for office-desk proactive care data
- best used as auxiliary multimodal emotion training, not as the main timing benchmark

### 5. CAER / CAER-S

Primary use:

- facial + scene/context-aware emotion robustness

Why use it:

- useful for hardening visual robustness to context
- fast to start compared with heavier licensed corpora

Primary source:

- official dataset page: [CAER](https://caer-dataset.github.io/)

Important note:

- it is more face+scene than true face+motion+audio
- therefore it is a supplemental visual robustness dataset, not the core tri-modal source

## Recommended Stage-2 Training Curriculum

### Stage A. Clean public pretraining for the tri-modal emotion encoder

Goal:

- replace "synthetic-only upstream perception" with a mixed public-data backbone

Recommended order:

1. face branch on Aff-Wild2
2. audio+video fusion on MAFW
3. motion alignment / finetuning on IEMOCAP
4. optional conversational refinement on MELD

Outputs to keep:

- fine-grained emotion logits
- valence
- arousal
- auxiliary stress / fatigue / suppression estimates

### Stage B. Tri-modal care relation finetuning

Goal:

- map multimodal affect + persona to:
  - need-care binary signal
  - `none / delay / immediate`
  - support intensity score

Training data:

- current synthetic care benchmark
- weaklabel transfer
- later real-pilot data

Important rule:

- public datasets should train the perception layer
- synthetic + weaklabel + pilot data should train the care-decision layer

### Stage C. Final proactive-care timing system

Goal:

- combine:
  - tri-modal encoder
  - care relation model
  - persona-aware timing model
  - constrained strategy output

The final system should not directly skip from public emotion data to claimed real proactive-care performance.

## Qualification Criteria

These are the gates I recommend using from now on.

### Gate 1. Upstream tri-modal emotion model is "qualified"

The model is only considered qualified if it satisfies all of:

1. On at least one clean public in-the-wild dataset:
   - fine-grained emotion Macro-F1 >= `0.50`

2. On a dimensional affect benchmark:
   - valence CCC or equivalent correlation >= `0.45`
   - arousal CCC or equivalent correlation >= `0.45`

3. On a second public dataset from a different source:
   - performance drop from in-domain to cross-dataset <= `15` absolute points in Macro-F1

4. Modal ablation behavior is sensible:
   - full fusion must beat every single modality
   - if face-only already equals fusion, the fusion design is not justified

5. Calibration:
   - ECE <= `0.08` on the public test set

Current status:

- not yet qualified by this gate, because current results are still synthetic-only

### Gate 2. Care relation model is "qualified"

The model is only considered qualified if it satisfies all of:

1. On synthetic care benchmark:
   - care-need AUROC >= `0.95`
   - care-timing Macro-F1 >= `0.85`

2. On held-out weaklabel data:
   - timing Macro-F1 >= `0.45`
   - balanced accuracy >= `0.50`

3. Calibration:
   - timing ECE <= `0.10`

4. False interruption control:
   - false-interrupt rate <= `0.20`

Current status:

- synthetic side: already strong
- weaklabel side: not yet qualified

### Gate 3. Full proactive-care system is "qualified"

The full system is only considered qualified if it satisfies all of:

1. real-pilot acceptance rate >= `0.60`
2. annoyance rate <= `0.15`
3. median response latency improves over rule baseline
4. timing balanced accuracy >= `0.55` on real-pilot labels
5. no single subgroup or persona slice collapses badly

Current status:

- not yet qualified
- blocked by missing real-pilot collection

## Practical Training Decision

If we optimize for the best real project outcome rather than the prettiest isolated benchmark, the correct next move is:

1. obtain `Aff-Wild2 + MAFW + IEMOCAP`
2. retrain the tri-modal encoder on clean public data
3. freeze or partially freeze it
4. finetune the care relation model on synthetic care data
5. transfer to weaklabel and later real-pilot data
6. evaluate against the gates above

## Honest Bottom Line

Right now we have:

- a strong synthetic tri-modal emotion model
- a strong synthetic tri-modal care relation model
- a useful but still weak downstream weaklabel timing system

That means the architecture direction is promising, but the project is not yet "qualified" in the sense that matters for deployment or a top-tier claim.
