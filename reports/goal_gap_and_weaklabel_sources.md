# Goal Gap And Weaklabel Sources Memo

## Scope
This memo is a read-only audit of how far the current `E:\Desktop\lunwen` research repository is from the previously stated user-document goals, and what historical artifacts in `E:\Desktop\emotion engine` and the user materials can be converted into weak labels.

Primary evidence reviewed:

- `E:\Desktop\gpt聊天文档\主动式情绪关怀系统工程实现方案深度调研与最优落地设计（中文）.docx`
- `E:\Desktop\gpt聊天文档\主动式情绪关怀机器人主动介入时机判断深度调研与落地报告.pdf`
- `E:\Desktop\gpt聊天文档\主动式情绪关怀机器人“主动介入时机判断”最优工程落地方案（端侧 Pi Zero2W + 电脑_边缘GPU A800）.pdf`
- `E:\Desktop\gpt聊天文档\4.1 与情绪识别博导支慧聪及其博士生的对话.txt`
- `E:\Desktop\gpt聊天文档\心理中心老师关于情绪关怀的建议.txt`
- `E:\Desktop\emotion engine`
- `E:\Desktop\lunwen`

## Executive Assessment
The current `lunwen` repository is a viable stage-1 research prototype, not yet the system described by the earlier documents. It has a usable schema, synthetic generator, weaklabel ingestion prototype, structured baseline, temporal neural baseline, joint placeholder, and CVPR-style paper scaffold. It does not yet satisfy the harder target stated across the user documents: a real video/persona conditioned model that predicts intervention timing and support strategy from multimodal observations, then validates the result with weak labels plus real pilot data.

The most important gap is not “missing polish”; it is that the current implementation still runs on synthetic or derived tabular-temporal features rather than actual video/audio streams and real human outcome labels.

## Priority Gap List

### P0. Real video / multimodal backbone is still missing
Target from user materials:

- The engineering documents repeatedly push a path of `video + audio + pose + persona + context -> timing`.
- The advisor conversation explicitly says the current task should not rely on single-modality rollback and should use multimodal historical state rather than only text or one modality.
- The deployment/material-selection doc pushes a real edge/cloud multimodal path: Pi-side sparse perception, A800-side sequence understanding, receptivity decision, and support generation.

Current `lunwen` state:

- `training/multitask/model.py` and `training/joint/model.py` currently train through `TemporalMultiHeadModel` and derived temporal arrays.
- `training/sequence_features.py` synthesizes temporal sequences from already-constructed observation/state/context fields.
- `training/video_backbone.py` contains a temporal conv module, but it is not consuming raw frames or real audio waveforms in the training path reviewed here.
- `paper/tables/audit_table.tex` still correctly marks the video backbone as pending.

Judgment:

- The repository does not yet implement the original target of a true video/multimodal backbone.
- It is closer to “temporal feature learner over synthetic observations” than to “video-to-intervention.”

### P0. Individual personality conditioning exists as schema, not as a validated modeling result
Target from user materials:

- The main goal is explicitly “emotion trend + personality -> whether and how to care.”
- The advisor conversation recommends personality-aware classification and even simplifying personality into a coarse inward/outward help preference when needed.
- The psychological advice stresses individual baseline, self-versus-self deviation, and dynamic confirmation rather than pure population-average judgment.

Current `lunwen` state:

- Persona fields exist in `training/schema.py` and engine contracts.
- Weaklabel generation derives pseudo Big Five style fields from logs in `training/weaklabel.py`.
- Temporal models accept persona arrays, but no ablation or paper evidence shows that persona conditioning materially improves timing decisions.
- No dynamic preference memory, no stable user embedding learned from real interactions, and no validated personalization calibration loop are reported.

Judgment:

- Personality is present at the API/data-schema level.
- Personality is not yet established as a real driver of model behavior with evidence.

### P0. Joint timing-to-strategy path is only a partial placeholder
Target from user materials:

- The intended target is not only `when` to intervene but also `how` to intervene.
- The advisor and psychological-center material both emphasize a staged strategy:
  - observe / describe
  - light inference
  - confirmation
  - correction / repair

Current `lunwen` state:

- `training/joint/model.py` predicts timing, strategy, and template class.
- The generated `CareUtteranceV1` remains template-like and not grounded in real support outcomes.
- There is no demonstrated learned mapping from video/persona to actual repairable support wording.
- No evaluation measures whether generated first-contact phrasing is accepted, rejected, annoying, or successfully repaired.

Judgment:

- The joint path exists as a code branch.
- It is not yet the user-targeted “video/personality directly output timing and support wording” model.
- It is also not yet aligned with the human-expert instruction that imperfect prediction must be recoverable through confirmation and repair.

### P0. Weaklabel is now feasible, but the current weaklabel branch is still too narrow
Target from user materials:

- The prior reports favor simulated data plus real logs plus later real pilot/MRT style learning.
- The materials explicitly distinguish state-layer data from intervention-layer data and recommend bootstrapping the intervention layer from product logs.

Current `lunwen` state:

- `datasets/weaklabel/manifest.json` shows a usable weaklabel set generated from `E:\Desktop\emotion engine\backend\auth.db`.
- Current weaklabel counts are:
  - train: 286
  - dev: 61
  - test: 62
- Current overall label distribution is not degenerate anymore:
  - timing: `none=160, immediate=187, delay=62`
  - strategy: `guard=94, observe=97, care=62, nudge=156`
- But current weaklabel extraction logic mainly reconstructs labels from event/message heuristics rather than from richer cross-source alignment.

Judgment:

- Weaklabel is not “missing”; it is now partially available.
- The gap is that it is still too shallow and too weakly validated to support top-tier claims.

### P0. Real pilot data is still absent from the reviewed evidence
Target from user materials:

- The documents consistently recommend simulated data first, then small real-user data, then personalized adjustment.
- The prior reports and discussions repeatedly say that end-to-end claims require self-collected real data.

Current `lunwen` state:

- `paper/sections/dataset_construction.tex` describes a real pilot plan.
- No reviewed artifact shows populated `datasets/realpilot`, real pilot metrics, or human preference outputs linked to the current models.
- `paper/sections/human_eval.tex` is still protocol text, not actual experiment evidence.

Judgment:

- Real-pilot calibration remains a hard gap.
- Without it, the paper can only claim synthetic and weaklabel bootstrapping, not validated proactive care quality.

### P1. The current paper is CVPR-shaped, but still far from top-tier standard
Top-tier gap by section:

- `abstract.tex`: still scaffold language; does not present a concrete dataset, method, or measured contribution.
- `introduction.tex`: contribution claims are still broad and repository-centered rather than paper-centered.
- `method.tex`: contains a placeholder figure and an idealized multimodal formulation that is ahead of the implementation actually reviewed.
- `experiments.tex`: states branches are pending; therefore the paper does not yet support its own method framing.
- `results_discussion.tex`: correctly cautious, but still reflects an earlier synthetic-only story and does not yet fully integrate the updated joint/weaklabel state.
- `human_eval.tex`: protocol only, no real study.
- `tables/main_results.tex` and `reports/formal_experiment_table.md` are inconsistent in scope and chronology.
- `reports/formal_experiment_table.md` contains impossible-looking `1.0000` weaklabel-test rows and `nan` entries, which are not top-tier publishable evidence.

Judgment:

- The repository has a paper system.
- It does not yet have a paper that meets a top-tier empirical standard.

### P1. Current experiment evidence is still closer to pipeline validation than scientific proof
Observed issues:

- Strongest timing result is still the structured baseline on synthetic data.
- Joint branch improves strategy metrics on synthetic labels but this is still synthetic and template-heavy.
- Weaklabel evaluation currently risks circularity because some labels are reconstructed from rule-triggered history that resembles the baseline policy.
- There is no error analysis by scene, workload, personality slice, or intervention cost.
- No calibration or false-interrupt analysis is reported, though the earlier materials frame “not disturbing the user” as central.

Judgment:

- The current repo validates training infrastructure and task plumbing.
- It does not yet prove the research thesis.

### P1. The human-expert interaction model is only partially reflected in the current system
What the human materials require:

- `心理中心老师关于情绪关怀的建议.txt` strongly emphasizes:
  - dynamic confirmation
  - white-description first
  - correction and repair
  - attention to privacy
  - non-diagnostic framing
  - self-versus-self baseline and long-term pattern
- `4.1 与情绪识别博导支慧聪及其博士生的对话.txt` emphasizes:
  - do not over-commit to end-to-end too early
  - use multimodal historical state
  - store emotion tables / trajectories rather than reprocessing raw history expensively
  - use synthetic plus small real data

Current `lunwen` state:

- The strategy schema mentions confirm/repair.
- The paper and model do not yet evaluate whether this interaction design actually reduces annoyance or recovers from wrong timing.

Judgment:

- The interaction philosophy is partially encoded.
- The evidence loop for it is still missing.

## Weaklabel Source Audit

### Source A. `E:\Desktop\emotion engine\backend\auth.db` is the strongest current weaklabel source
Evidence:

- Database tables exist and are populated.
- `emotion_events` schema:
  - `user_id, timestamp_ms, type, description, v, a, t, s, intensity, source`
- `chat_messages` schema:
  - `user_id, sender, text, timestamp_ms, content_type, attachments_json`
- Counts observed:
  - `emotion_events = 656`
  - `chat_messages = 2838`

Why it is valuable:

- It can support timestamp-aligned reconstruction of:
  - trigger times
  - pre/post bot intervention
  - user response latency
  - rejection/acceptance heuristics
  - derived persona and preference signals

Limitations:

- Labels are indirect.
- Message sentiment and acceptance still need better parsing than current keyword heuristics.

### Source B. `E:\Desktop\emotion engine\logs\bridge*.log` is the richest event-level weaklabel source
Evidence:

- Files such as `bridge.out.log`, `bridge_latest.log`, `bridge_live.log`, `bridge_wake_1772062794.out.log` contain repeated structured records:
  - `[TriggerFired] {...}`
  - `[CarePlanReady] {...}`
- `CarePlanReady` entries include:
  - `care_plan.decision`
  - `care_plan.level`
  - `care_plan.cooldown_min`
  - `care_plan.followup_question`
  - `delivery_mode`
  - `reason.pattern`
  - `reason.tags`
  - `detail.V_sub`
  - `detail.A_sub`
  - `detail.T_sub`

Why it is valuable:

- This is almost a weakly supervised intervention record already.
- It can be converted into:
  - timing labels
  - strategy labels
  - delivery-mode labels
  - pseudo state vectors
  - intervention rationales
  - rule-fired positive windows and skipped windows

Limitations:

- Many records are rule-generated, so they are not human gold labels.
- Some logs contain encoding corruption in Chinese text, which affects direct language analysis but not most structural fields.

### Source C. Emotion history and realtime-detail APIs indicate recoverable temporal curves
Evidence:

- `emotion engine` logs repeatedly show:
  - `GET /api/emotion/realtime/detail`
  - `GET /api/emotion/history?limit=50`
  - in some cases `GET /api/emotion/history?limit=600&start_ms=...&end_ms=...`
- `emotion engine/app windows/services/emotionService.ts` directly calls:
  - `/api/emotion/realtime/detail`
  - `/api/emotion/history?limit=50`
  - `/api/emotion/history?...`

Why it is valuable:

- This strongly suggests the product already exposes short-term and longer-term emotional trajectories.
- Those trajectories can become weak supervision for:
  - trend slope
  - volatility
  - sustained low/high states
  - pre-intervention context features

Limitations:

- The raw response payload was not directly dumped in this audit.
- The interface exists; payload extraction still needs one more pass.

### Source D. `CarePlanReady` plus downstream user chat is the best current proxy for acceptance/annoyance
Evidence:

- The current weaklabel builder already joins `emotion_events` and `chat_messages`.
- The expert materials repeatedly say wrong interventions should be repairable and should be tracked via user response.
- Current app/backend clearly support chat exchange after interventions.

Why it is valuable:

- It can produce weak outcome labels:
  - accepted
  - ignored
  - annoyed
  - response latency
  - follow-up engagement length

Limitations:

- Requires better Chinese response parsing and better conversation-window attribution.

### Source E. Manual-care and policy-control events are available in code and should be harvested
Evidence in code:

- `engine/core/engine_controller.py` contains:
  - `manual_care`
  - `CarePlanSkipped`
  - `CarePlanReady`
  - `cooldown_min`
  - `daily_limit`
  - `followup_question`
  - `record_event`

Why it is valuable:

- These control-path events provide negative and counterfactual-ish supervision:
  - skipped due to cooldown
  - manual interventions
  - policy-blocked interventions
  - modified intervention parameters

Limitations:

- These sources were confirmed in code, but not yet extracted into the current `lunwen` weaklabel pipeline.

### Source F. `server_backend/data/auth.db` is structurally useful but currently empty
Evidence:

- The same tables exist there, but both `emotion_events` and `chat_messages` were observed with count `0`.

Judgment:

- Keep it as a future ingestion location.
- Do not treat it as an active source for the current paper.

## Distance To Original Goal
Using the user documents as the target definition, the current repository is approximately at:

- close on task decomposition and engineering interfaces
- moderate on synthetic bootstrapping and rule-to-ML transition
- weak on true multimodal modeling
- weak on validated personality conditioning
- weak on joint timing-to-strategy generation
- moderate on weaklabel feasibility
- very weak on real-pilot evidence
- weak on top-tier paper evidence standards

In short: `lunwen` is already a serious prototype repository, but still materially short of the original ambition described in the earlier documents.

## Recommended Next Read-Only Conclusions
1. Treat the current strongest weaklabel source as `backend/auth.db + bridge*.log`, not `server_backend/data/auth.db`.
2. Do not describe the current neural models as true video models; they are temporal models over engineered or derived features.
3. Keep the paper’s main claim centered on a new task/data/problem framing until real multimodal and real-pilot evidence exists.
4. Prioritize extracting:
   - `CarePlanReady`
   - `CarePlanSkipped`
   - `TriggerFired`
   - emotion-history windows
   - user chat replies after intervention
5. Require future experiments to report:
   - false interrupt rate
   - annoyance / acceptance
   - calibration
   - user/persona slice analysis
   - error recovery after wrong first contact
