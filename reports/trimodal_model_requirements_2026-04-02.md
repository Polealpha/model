# Tri-Modal Model Requirements 2026-04-02

## 1. Distilled Requirements From The Transcript

The transcript does not argue for replacing the current proactive-care timing task. It argues for splitting the full problem into a cleaner pipeline and making the perception layer materially stronger.

The concrete requirements I extract are:

1. The project should not rely on a single generic "emotion recognition" score.
   The conversation repeatedly separates:
   - `what emotion / how strong`
   - `does the person currently need external care`
   - `if yes, what kind of intervention`

2. A new perception model should perform fine-grained emotion understanding from three observable modalities:
   - face / facial affect
   - action / movement / posture / body behavior
   - pure audio

3. The tri-modal model should not stop at coarse positive-vs-negative recognition.
   The transcript explicitly points toward:
   - fine-grained categories
   - emotion intensity / amplitude
   - valence / arousal style continuous signals
   - temporal fluctuation rather than a single frame decision

4. Personality should remain an explicit input, not an emergent side effect.
   The transcript treats personality as a known or inferable upstream signal:
   - initial persona profile can come from dialogue-based profiling
   - a coarse class such as "more likely to seek outside help" vs "more likely to cope inwardly" is already useful
   - timing should condition on both emotion trajectory and personality

5. The project should avoid a naive end-to-end claim at this stage.
   The transcript is very clear on this point:
   - there is no standard dataset for "active care timing"
   - a full end-to-end model is hard to justify immediately
   - first validate the decomposed workflow
   - rules / decision tree remain acceptable as a fallback layer

6. The correct scientific direction is:
   - public emotion datasets for perception pretraining
   - simulated or controlled data for timing supervision
   - a small amount of real data for calibration
   - later iterative refinement from product feedback

7. The transcript explicitly recommends psychologically grounded targets.
   The most important ones mentioned are:
   - valence
   - arousal
   - emotion amplitude / fluctuation
   - whether the user is currently receptive to outside intervention

8. Wearables are optional, not core.
   The discussion mentions watches / wristbands as potentially useful, but not necessary for the first tri-modal model. The first useful version should already work from camera + audio + motion-derived cues.

9. The first-stage goal is feasibility and evidence, not theoretical perfection.
   The transcript repeatedly endorses:
   - simulated data
   - controlled collection
   - small-group internal testing
   - building a usable first version before claiming generality

## 2. How This Differs From The Current Repository

The current repository is still a temporal descriptor prototype, not a raw tri-modal perception system.

### Current implementation in the repo

Relevant files:

- `E:\Desktop\lunwen\training\temporal_backbone.py`
- `E:\Desktop\lunwen\training\multitask\model.py`
- `E:\Desktop\lunwen\training\joint\model.py`
- `E:\Desktop\lunwen\training\weaklabel.py`
- `E:\Desktop\lunwen\training\schema.py`
- `E:\Desktop\lunwen\paper\sections\method.tex`

What the repo currently does:

- uses a synthetic or weaklabel `EpisodeWindowV1` schema
- consumes pre-derived observation descriptors such as:
  - `face_presence_ratio`
  - `gaze_avert_ratio`
  - `head_motion_var`
  - `posture_slouch_score`
  - `voice_energy`
  - `prosody_stress`
- fuses those descriptors with persona and context
- predicts:
  - timing
  - latent state
  - strategy/template

### What is still missing relative to the transcript's new requirement

1. No raw face stream encoder.
   There is no branch that ingests face crops or raw image/video tokens.

2. No explicit action / body branch.
   Current motion cues are scalar proxies, not a dedicated pose or body-action model.

3. No pure audio encoder.
   Audio is represented only through scalar features like silence / energy / stress proxy.

4. No fine-grained emotion taxonomy head.
   The repo predicts six abstract state dimensions in `STATE_LABELS`, but not a richer emotion label space.

5. No explicit valence-arousal supervision branch tied to tri-modal perception.
   Valence / arousal exist in the current state vector, but they are still downstream synthetic fields rather than outputs of a proper tri-modal encoder trained on emotion data.

6. No public-dataset pretraining stage for the perception model.
   The current pipeline is mostly synthetic plus weaklabel.

7. No cross-domain / multi-task design for perception-to-timing transfer.
   The transcript specifically points toward multi-task or cross-domain learning; the repo does not yet implement that for raw modalities.

8. No controlled data-collection workflow dedicated to tri-modal fine-grained labels.
   The current software and dataset path are not yet specialized for collecting face + action + audio emotion annotations or self-reported receptivity.

## 3. Recommended Model / Data / Training Setup For Dual A800

The cleanest next-step architecture is not to replace the current timing model outright. It is to insert a proper tri-modal perception model ahead of it.

### 3.1 Target decomposition

Recommended decomposition:

1. `Tri-Modal Emotion Encoder`
   Input:
   - face video
   - pose / action sequence
   - audio

   Output:
   - fine-grained emotion class
   - valence
   - arousal
   - optional dominance / stress / fatigue auxiliary scores

2. `Emotion-Trajectory + Persona Receptivity Model`
   Input:
   - tri-modal emotion outputs over time
   - persona profile
   - context flags

   Output:
   - `none / delay / immediate`
   - optional strategy prior

3. `Strategy / utterance layer`
   Keep the current constrained strategy layer for now instead of claiming open-ended generation is solved.

This is much closer to what the transcript recommends than a single monolithic end-to-end model.

### 3.2 Recommended tri-modal architecture

Use a three-branch model with late or mid-level fusion.

#### Face branch

Recommended options:

- `VideoMAE`-style face clip encoder
- `Swin/ViT`-style short video backbone over aligned face crops
- if implementation pressure is high: frame encoder + temporal transformer

Input suggestion:

- 16 to 32 frames per clip
- aligned face crop at `224x224`
- clip duration `2s to 4s`

Targets:

- fine-grained emotion class
- valence / arousal regression

#### Action branch

Recommended options:

- `ST-GCN` over 2D keypoints
- `PoseC3D`
- lightweight temporal MLP/Transformer over extracted pose trajectories if collection is weak

Input suggestion:

- 17 or 33 keypoints
- 32 to 64 timesteps
- derived posture and movement velocity features can be auxiliary inputs

Targets:

- body-expression emotion cues
- arousal / activation intensity
- agitation / withdrawal style auxiliaries

#### Audio branch

Recommended options:

- log-mel spectrogram encoder with AST / CNN-T
- wav2vec2 / HuBERT style encoder if enough engineering time exists
- practical first step: spectrogram CNN + transformer pooling

Input suggestion:

- `4s to 6s` raw audio
- `16kHz`
- 80-bin log-mel features

Targets:

- fine-grained affect class
- valence / arousal
- stress / tension auxiliary regression

#### Fusion block

Recommended options:

- cross-attention transformer over modality tokens
- gated fusion with modality dropout
- missing-modality robustness built in from the start

Final heads:

- `emotion_class_head`
- `valence_reg_head`
- `arousal_reg_head`
- optional `stress_reg_head`
- optional `fatigue_reg_head`

### 3.3 Recommended label space

Do not restrict the perception model to only three or four broad emotions.

Recommended output structure:

1. Discrete fine-grained class head:
   - neutral
   - calm
   - joy
   - relief
   - sadness
   - loneliness
   - anxiety
   - stress
   - frustration
   - anger
   - overwhelm
   - fatigue

2. Continuous affect heads:
   - valence
   - arousal

3. Optional auxiliary heads:
   - social-withdrawal tendency
   - help-seeking tendency
   - externalization vs internalization tendency

Important note:
`help-seeking tendency` should not be learned from emotion alone. It should be fused with persona and longer history in the downstream receptivity model.

### 3.4 Recommended data strategy

The transcript strongly supports a staged data plan. The right version for this repo is:

#### Stage A: public pretraining for tri-modal emotion perception

Use public datasets to learn the tri-modal encoder or its unimodal branches.

Reasonable dataset mix:

- face / video emotion datasets
- body-language / posture / affect datasets
- audio emotion recognition datasets

The exact dataset list should be selected by license and label compatibility, but the principle is clear:
- public data for emotion perception
- self/controlled data for receptivity
- real product feedback for calibration

#### Stage B: controlled internal collection

The transcript explicitly suggests:

- controlled emotion elicitation
- team volunteers
- self-report of whether intervention would have been wanted during specific periods

That gives a small but valuable dataset for:

- emotion fine-grained validation
- timing preference labels
- persona-conditioned receptivity

#### Stage C: simulated timing labels

Keep the current synthetic route, but upgrade it:

- tri-modal signals should be generated or weakly simulated per modality
- not only scalar abstract state curves
- include emotion amplitude and fluctuation
- include personality-conditioned preference

#### Stage D: weaklabel and real-product calibration

Continue using:

- `auth.db`
- bridge logs
- downstream user replies

but now use them to calibrate the second-stage receptivity model rather than trying to treat them as full tri-modal gold data.

### 3.5 Recommended training curriculum on dual A800

The dual-A800 setup is enough for a real step up, but only if training is staged.

#### Curriculum

1. Train unimodal encoders separately.
   - face branch
   - action branch
   - audio branch

2. Warm-start the tri-modal fusion model on fine-grained emotion + valence/arousal.

3. Freeze or partially freeze lower encoders.

4. Train the downstream receptivity model on:
   - synthetic trajectory data
   - weaklabel timing data
   - controlled internal collection

5. Fine-tune fusion end-to-end only after the downstream task is stable.

#### Practical A800 configuration

Recommended starting point:

- 2 GPUs with DDP or `DataParallel`
- mixed precision enabled
- gradient accumulation if clip length is increased
- modality dropout during training
- balanced sampling by emotion class
- focal loss or class-balanced CE for rare emotion classes
- CCC / MSE for valence-arousal regression

Suggested first heavy run:

- face clip: `16 frames`, `224x224`
- pose steps: `48`
- audio: `4s to 6s`
- hidden size: `384 to 768`
- fusion layers: `4 to 6`
- batch size per GPU:
  - `8 to 16` for full raw tri-modal clips
  - larger if pose/audio are pre-extracted

### 3.6 Evaluation plan

The new tri-modal model should be evaluated at two levels.

#### Perception-level evaluation

- fine-grained emotion macro-F1
- valence CCC / MAE
- arousal CCC / MAE
- modality ablation:
  - face only
  - action only
  - audio only
  - face + action
  - face + audio
  - action + audio
  - all three

#### Timing-level evaluation

- replace current descriptor state inputs with tri-modal outputs
- measure timing balanced accuracy / Macro-F1
- compare:
  - current descriptor prototype
  - tri-modal-perception + timing head
  - tri-modal-perception + timing + persona

This is the cleanest way to prove the new model actually improves proactive-care timing rather than only making the emotion branch look more sophisticated.

## 4. What Can Be Implemented Immediately In This Repo This Round

The current round should not attempt the full ideal system all at once. The highest-yield immediate work is:

### 4.1 Add a dedicated tri-modal perception package

Add a new module family under `training/trimodal/` with:

- `datasets.py`
- `encoders_face.py`
- `encoders_pose.py`
- `encoders_audio.py`
- `fusion.py`
- `model.py`
- `losses.py`

This can coexist with the current `multitask/` and `joint/` branches.

### 4.2 Extend the schema instead of replacing it

Upgrade `E:\Desktop\lunwen\training\schema.py` so `EpisodeWindowV1` can optionally include:

- face clip metadata
- pose sequence path or keypoint array path
- audio waveform path
- fine-grained emotion label
- valence / arousal annotations
- modality availability flags

Do not break the current synthetic and weaklabel pipelines.

### 4.3 Build a perception-first synthetic upgrade

Before real public-dataset integration is complete, upgrade the synthetic generator so each episode contains:

- face-affect proxy sequence
- body-action proxy sequence
- audio-affect proxy sequence
- fine-grained emotion class
- valence / arousal

This gives the new model a compatible training interface immediately.

### 4.4 Train a first tri-modal proxy model this round

This round can realistically implement:

1. tri-modal descriptor branches
   - one branch for face-affect sequence
   - one branch for pose/action sequence
   - one branch for audio-affect sequence

2. multi-task outputs
   - fine-grained emotion class
   - valence
   - arousal
   - optional stress/fatigue

3. export of emotion trajectory into the existing timing model

This is not yet raw-video end to end, but it is a meaningful bridge from the current prototype toward the transcript's target.

### 4.5 Add a real data collection path in the software

Even before perfect modeling, the repo should add a collection path for:

- camera face clips
- pose/keypoint traces
- short audio clips
- self-report emotion label
- self-report "would you want intervention right now"

This is the most important non-model addition if the team wants the next iteration to stop depending almost entirely on synthetic data.

### 4.6 Keep the timing layer decomposed

This round should still preserve:

- a rule fallback
- a structured baseline
- a learned downstream timing model

This is fully aligned with the transcript's caution against pretending the full end-to-end problem is already solved.

## 5. Recommended Immediate Roadmap

### Step 1

Implement a `trimodal_proxy` branch inside the current repo:

- synthetic tri-modal descriptors
- emotion fine-grained classification
- valence / arousal regression
- export to timing model

### Step 2

Run dual-A800 training on:

- synthetic-only tri-modal perception
- synthetic + weaklabel timing downstream
- perception ablations by modality

### Step 3

Wire the desktop software to collect:

- face
- pose
- audio
- self-report timing preference

### Step 4

Start replacing proxy inputs with actual pre-extracted:

- face clips
- keypoints
- log-mel features

### Step 5

Only after the above is stable, consider claiming a real tri-modal end-to-end path in the paper.

## 6. Bottom Line

The transcript points to a very specific next move:

- keep the proactive-care timing task
- strengthen the perception layer with face + action + audio
- predict fine-grained emotion plus valence/arousal
- fuse that with personality for receptivity
- do not overclaim end-to-end readiness yet

The current repo is close enough to support this pivot immediately, but only if the team treats the tri-modal model as a new perception module feeding the timing system, not as a magical one-shot replacement for the whole pipeline.
