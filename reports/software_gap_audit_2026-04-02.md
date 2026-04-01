# Software Gap Audit 2026-04-02

## 1. Current Implemented Surfaces

The desktop app already provides a substantial product shell. The main implemented surfaces are:

### 1.1 Product-facing desktop shell

- Main app shell with dashboard, chat, persona, device, control, and profile tabs:
  - `E:\Desktop\lunwen\app\desktop\App.tsx`
- Native Electron entrypoints, overlay window, tray, and local backend bootstrap:
  - `E:\Desktop\lunwen\app\desktop\electron\main.cjs`
  - `E:\Desktop\lunwen\app\desktop\electron\preload.cjs`

### 1.2 Auth / activation / personality onboarding

- Login, activation gate, owner binding, and assessment flow are present:
  - `E:\Desktop\lunwen\app\desktop\components\ActivationGate.tsx`
  - `E:\Desktop\lunwen\app\desktop\services\activationService.ts`
- The app already exposes a text-based personality and care-preference assessment surface:
  - `E:\Desktop\lunwen\app\desktop\services\activationService.ts:25`
  - `E:\Desktop\lunwen\app\desktop\services\activationService.ts:33`
  - `E:\Desktop\lunwen\app\desktop\components\CompanionProfilePanel.tsx:107`

### 1.3 Emotion / event / care-plan visualization

- Realtime risk polling and history retrieval exist:
  - `E:\Desktop\lunwen\app\desktop\services\emotionService.ts:4`
  - `E:\Desktop\lunwen\app\desktop\services\emotionService.ts:22`
  - `E:\Desktop\lunwen\app\desktop\services\emotionService.ts:26`
- WebSocket event ingestion exists:
  - `E:\Desktop\lunwen\app\desktop\services\eventService.ts:12`
- The app already reacts to product-side proactive events such as `TriggerFired` and `CarePlanReady`:
  - `E:\Desktop\lunwen\app\desktop\App.tsx:1410`
  - `E:\Desktop\lunwen\app\desktop\App.tsx:1425`

### 1.4 Chat and reminder interaction

- There is a mature desktop chat surface with attachments, voice capture, and assistant runtime checks:
  - `E:\Desktop\lunwen\app\desktop\components\ChatInterface.tsx`
- There is assistant runtime polling and due-reminder polling:
  - `E:\Desktop\lunwen\app\desktop\App.tsx:73`

### 1.5 Device control and settings

- Device settings, provisioning, and device monitoring are present:
  - `E:\Desktop\lunwen\app\desktop\services\deviceService.ts`
  - `E:\Desktop\lunwen\app\desktop\components\DeviceMonitor.tsx`
  - `E:\Desktop\lunwen\app\desktop\components\Provisioning.tsx`
  - `E:\Desktop\lunwen\app\desktop\components\SettingsPanel.tsx`

### 1.6 Research engine contracts and rule pipeline

- The repo already has a clean research-engine contract layer:
  - `E:\Desktop\lunwen\engine\schemas\contracts.py`
- The repo already has a structured timing -> strategy -> utterance pipeline:
  - `E:\Desktop\lunwen\engine\dialogue\pipeline.py`
  - `E:\Desktop\lunwen\engine\policy\rule_baseline.py`
  - `E:\Desktop\lunwen\engine\state\estimator.py`

## 2. Missing Integration Points Between App and Engine

The biggest issue is not lack of UI. It is that the desktop app and the research engine still live in parallel instead of as one system.

### 2.1 No actual bridge from app observations to `EpisodeWindowV1`

The research engine expects a structured sample:

- `EpisodeWindowV1`
- `PersonaProfileV1`
- `ContextFlagsV1`
- `StateVectorV1`

These are defined in:

- `E:\Desktop\lunwen\engine\schemas\contracts.py`

But the app never constructs or sends that schema. There is no path from:

- realtime face/audio/risk state
- device mode
- personality assessment
- chat context

into an `EpisodeWindowV1` inference request.

Current app transport only supports:

- `/api/emotion/*`
- `/api/device/*`
- `/api/assistant/*`
- `/api/engine/signal`
- `ws/events`

Relevant files:

- `E:\Desktop\lunwen\app\desktop\README.md:42`
- `E:\Desktop\lunwen\app\desktop\services\engineService.ts:15`

There is no research inference API such as:

- `/api/research/infer`
- `/api/research/feedback`
- `/ws/research`

### 2.2 App types and engine contracts are semantically mismatched

The desktop app uses:

- coarse `EmotionType`
- `RiskScores {V,A,T,S}`
- `CarePlan {text, style, motion, emo, followup_question}`

Defined in:

- `E:\Desktop\lunwen\app\desktop\types.ts`

The research engine uses:

- `TimingDecisionV1`
- `StrategyPlanV1`
- `CareUtteranceV1`
- `FeedbackEventV1`

Defined in:

- `E:\Desktop\lunwen\engine\schemas\contracts.py`

These type systems do not match. The desktop app has no first-class surface for:

- timing class
- timing confidence
- uncertainty
- why codes
- strategy level
- utterance constraints
- structured feedback outcome

That means the paper’s main objects do not exist on the software side.

### 2.3 The app only accepts product-style `CarePlanReady`, not research-model outputs

The current event handler only surfaces care plans when the payload looks like the older product backend shape. It also filters by `content_source`:

- `E:\Desktop\lunwen\app\desktop\App.tsx:1425`

The code explicitly ignores plans unless:

- `content_source === "llm"`
- or a very specific manual fallback path is hit

So even if the research engine produced a rule-based or learned structured output, the app would currently drop it unless the backend spoofed the older LLM-oriented payload format.

This is a core integration blocker.

### 2.4 Personality onboarding exists, but it is not mapped into engine persona vectors

The app already captures textual personality / preference structure:

- `interaction_preferences`
- `decision_style`
- `stress_response`
- `comfort_preferences`
- `avoid_patterns`
- `care_guidance`

See:

- `E:\Desktop\lunwen\app\desktop\services\activationService.ts:48`
- `E:\Desktop\lunwen\app\desktop\components\CompanionProfilePanel.tsx:107`

But the research engine expects:

- `PersonaProfileV1.big_five`
- `PersonaProfileV1.interaction_preferences`

There is no normalization layer from activation/personality outputs into the engine’s numeric persona profile. The app collects personality, but the model path does not consume it in a frozen way.

### 2.5 No software-side feedback capture for paper-critical outcomes

The research contracts include:

- `FeedbackEventV1.accepted`
- `FeedbackEventV1.ignored`
- `FeedbackEventV1.annoyed`
- `FeedbackEventV1.response_latency_ms`

Defined in:

- `E:\Desktop\lunwen\engine\schemas\contracts.py`

But the desktop app has no explicit UI or persistence path for these outcomes. Search results show no real feedback controls for:

- accepted
- ignored
- annoyed
- response latency

The current app can show a popup and append a bot message, but it cannot produce the feedback labels needed to close the loop for real-pilot data.

### 2.6 No software path for real-pilot dataset capture

The paper and repo require eventual real-pilot collection, but the app currently does not appear to support:

- saving short consented observation windows
- exporting aligned multimodal samples
- attaching personality snapshot + context flags + intervention result
- writing structured pilot records to `datasets/realpilot`

This means the software is not yet acting as a research instrument.

### 2.7 Device / risk surfaces are richer than the engine actually receives

The app exposes detailed `riskDetail` fields, including expression IDs, fatigue, attention drop, silence, RMS, and other signals:

- `E:\Desktop\lunwen\app\desktop\components\DeviceMonitor.tsx:406`

But there is no explicit mapping from these app-visible realtime signals into:

- `feature_vector`
- `context_flags`
- `StateVectorV1`
- `EpisodeWindowV1`

So the desktop side already has useful telemetry, but the research engine is not formally wired to consume it.

## 3. Highest-Impact Changes To Make the Desktop App Usable for This Paper Prototype

### 3.1 Add a research inference gateway and unify the event schema

Highest impact change:

- expose a backend endpoint that accepts an `EpisodeWindowV1`-compatible payload
- return `timing_decision`, `strategy_plan`, and `care_utterance`
- broadcast a research event over WebSocket

Recommended event contract:

- `ResearchDecisionReady`
- payload:
  - `timing_decision`
  - `strategy_plan`
  - `care_utterance`
  - `state_vector`
  - `model_name`
  - `model_version`
  - `source_window_id`

This lets the app render the research system directly instead of depending on legacy `CarePlanReady`.

### 3.2 Add an app-side `EpisodeWindow` builder

The desktop app needs a deterministic sample builder that assembles:

- persona snapshot from activation/profile state
- context flags from mode/device settings
- feature vector from `riskDetail`, wake state, speaking state, and device activity
- optional references to cached image/audio snippets

Without this builder, the research model remains disconnected from the live app.

### 3.3 Replace the current popup-only care UX with a structured decision panel

The current app can show a popup and inject a bot message, but the paper prototype needs the operator-facing state visible.

Add a panel that shows:

- timing decision: `immediate / delay / none`
- confidence / uncertainty
- why codes
- selected strategy
- proposed first utterance
- fallback / repair text

This is the minimum software surface that matches the paper.

### 3.4 Add explicit intervention-outcome buttons and persistence

The desktop app needs direct feedback capture for every proactive intervention:

- helpful / accepted
- ignored
- annoying / mistimed
- not now

Also record:

- response latency
- whether repair text was used
- whether the user switched to chat

This is the most important missing data loop for real-pilot quality.

### 3.5 Normalize activation/personality output into engine persona features

The personality assessment is already a strong asset. It needs a formal adapter into:

- `big_five` or a reduced numeric persona vector
- interruption tolerance
- care seeking
- privacy sensitivity
- preferred contact style

This is lower effort than building the full raw-video stack, and it will immediately make the software side more faithful to the paper.

### 3.6 Add a research-mode runtime toggle

Right now the desktop app behaves like a product client talking to a legacy backend. Add an explicit runtime mode:

- product mode
- research timing mode
- research joint mode

This would make it possible to demo and evaluate the paper system without breaking existing product flows.

### 3.7 Make the live view honest

`LiveView.tsx` currently shows a stock image instead of an actual camera stream:

- `E:\Desktop\lunwen\app\desktop\components\LiveView.tsx`

For a research prototype, either:

- wire in the real stream
- or clearly mark it as placeholder and keep it out of the evaluation path

Otherwise the software demo implies capabilities that are not actually connected.

## 4. Obvious Broken Assumptions in the Current App

### 4.1 The app assumes a backend that already knows how to be the product

The desktop app is wired around a mature product-style backend contract, while the research engine is currently just a Python package plus training code. The repo does not yet contain the service layer that turns the research engine into the backend the app expects.

This is the biggest broken assumption in the current system.

### 4.2 The app assumes care content is mainly LLM-authored

The `CarePlanReady` handling in `App.tsx` filters on LLM-oriented metadata. That is incompatible with a research setup where:

- rule models
- structured models
- joint timing/strategy models

should all be first-class sources.

### 4.3 The app assumes coarse emotion categories are enough

The app’s top-level `EmotionType` is still only:

- `HAPPY`
- `SAD`
- `ANGRY`
- `CALM`
- `TIRED`
- `ANXIOUS`

See:

- `E:\Desktop\lunwen\app\desktop\types.ts:1`

That is already below the paper’s current state/timing/strategy representation, and it is far below the next tri-modal fine-grained emotion direction.

### 4.4 The app does not yet behave like a data-collection instrument

For the paper roadmap, the desktop client should help collect:

- structured pilot windows
- intervention outcomes
- personality snapshots
- operator-visible model explanations

Right now it is still mainly a product UI, not a research collection and evaluation surface.

### 4.5 There are visible encoding / text integrity problems

Some files contain mojibake or corrupted Chinese strings, including parts of:

- `E:\Desktop\lunwen\app\desktop\electron\main.cjs`
- `E:\Desktop\lunwen\engine\policy\rule_baseline.py`

This is not just cosmetic. It creates risk for:

- poor demo quality
- unreadable care utterances
- broken operator understanding during pilot use

## Bottom Line

The desktop app is already a strong shell, but it is still a shell around the old product contract. The research engine is already a usable prototype, but it is still mostly offline and package-level. The missing piece is a real integration layer:

1. build `EpisodeWindow` from live app state
2. call the research model through a backend inference gateway
3. render timing / strategy / utterance explicitly
4. capture user outcome feedback
5. persist pilot data for the next training cycle

Until that bridge exists, the paper prototype and the software prototype remain adjacent rather than unified.
