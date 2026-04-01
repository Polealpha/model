# Labeling Specification

## EpisodeWindowV1

Each sample is a 60-second observation window with a 5-second stride.

## State labels

- `valence`: current pleasantness estimate, not a diagnosis.
- `arousal`: activation level.
- `stress`: short-horizon overload proxy.
- `fatigue`: low-energy proxy.
- `attention_drop`: current engagement decline proxy.
- `suppression`: mismatch between observed expression and likely internal pressure.

## Timing labels

- `immediate`: a proactive check-in is justified now.
- `delay`: likely useful later, but the current moment is not ideal.
- `none`: do not proactively interrupt now.

## Strategy labels

- `observe`: silent monitoring only.
- `nudge`: lightweight deferred touchpoint.
- `care`: proactive caring check-in.
- `guard`: stronger supportive check-in with repair path.

## Annotation rules

1. Never use medical diagnosis labels in v1.
2. Distinguish `need_support` from `ready_for_support`.
3. Prefer `delay` over `immediate` when `busy_speaking`, privacy risk, or repeated interruption is present.
4. Use `guard` only when vulnerability is high and the utterance includes an explicit repair option.

