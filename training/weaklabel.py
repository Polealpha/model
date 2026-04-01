from __future__ import annotations

import ast
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .data import save_episodes
from .io import save_json
from .schema import EpisodeWindowV1


NEGATIVE_REPLY_MARKERS = ("别", "不要", "烦", "算了", "闭嘴", "不用", "滚")
CARE_MARKERS = ("陪", "聊聊", "说说", "不想多说也没关系", "我在这儿", "慢慢梳理")
NUDGE_MARKERS = ("要不要", "歇会", "喝口水", "活动一下", "缓一缓", "先休息")


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _contains_any(text: str, patterns: Sequence[str]) -> bool:
    return any(pattern in text for pattern in patterns)


@dataclass
class WeakLabelConfig:
    window_ms: int = 60000
    reply_window_ms: int = 10 * 60 * 1000
    recent_window_ms: int = 60 * 60 * 1000
    bridge_glob: str = "bridge*.log"
    include_bridge: bool = False


def _load_rows(conn: sqlite3.Connection, query: str, params: Tuple[Any, ...] = ()) -> List[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute(query, params)
    return cur.fetchall()


def _derive_persona(user_events: Sequence[sqlite3.Row], user_messages: Sequence[sqlite3.Row]) -> Dict[str, float]:
    trigger_count = sum(1 for row in user_events if row["description"] == "event:TriggerFired")
    reply_count = sum(1 for row in user_messages if row["sender"] == "user")
    bot_count = sum(1 for row in user_messages if row["sender"] == "bot")
    avg_user_len = (sum(len(row["text"] or "") for row in user_messages if row["sender"] == "user") / max(1, reply_count))
    avg_intensity = sum(int(row["intensity"] or 0) for row in user_events) / max(1, len(user_events))
    return {
        "big5_openness": _clamp01(0.45 + 0.01 * min(20.0, avg_user_len) / 20.0),
        "big5_conscientiousness": _clamp01(0.55 - 0.08 * min(1.0, trigger_count / 50.0)),
        "big5_extraversion": _clamp01(0.35 + 0.20 * min(1.0, reply_count / max(1, bot_count + 1))),
        "big5_agreeableness": _clamp01(0.55 + 0.10 * min(1.0, reply_count / max(1, trigger_count + 1))),
        "big5_neuroticism": _clamp01(0.35 + 0.25 * min(1.0, avg_intensity / 20.0) + 0.10 * min(1.0, trigger_count / 40.0)),
        "pref_directness": _clamp01(0.55 + 0.10 * min(1.0, avg_user_len / 40.0)),
        "pref_checkin_frequency": _clamp01(0.35 + 0.20 * min(1.0, trigger_count / 30.0)),
        "pref_touchiness": _clamp01(0.20 + 0.10 * min(1.0, avg_intensity / 20.0)),
    }


def _derive_context(timestamp_ms: int, daily_count: int, cooldown_active: bool, recent_accepts: int, recent_rejections: int, message_text: str) -> Dict[str, float]:
    hour = ((timestamp_ms // 1000) // 3600) % 24
    work_hour = 9 <= hour <= 18
    busy_by_text = _contains_any(message_text, ("会议", "工位", "进度", "延迟", "上班"))
    return {
        "busy_speaking": float(work_hour and busy_by_text),
        "privacy_on": float(_contains_any(message_text, ("隐私", "别看", "别动"))),
        "quiet_mode": float(hour >= 23 or hour <= 6),
        "daily_count": float(daily_count),
        "cooldown_active": float(cooldown_active),
        "meeting_pressure": float(0.65 if busy_by_text else 0.25 if work_hour else 0.10),
        "social_load": float(0.20 if hour <= 8 else 0.45 if work_hour else 0.35),
        "time_of_day": float(hour / 23.0 if hour else 0.0),
        "recent_rejections": float(recent_rejections),
        "recent_accepts": float(recent_accepts),
    }


def _derive_state(event: sqlite3.Row, support_text: str, context_flags: Dict[str, float]) -> Dict[str, float]:
    intensity = float(event["intensity"] or 0.0)
    v = float(event["v"] or 0.0)
    a = float(event["a"] or 0.0)
    t = float(event["t"] or 0.0)
    s = float(event["s"] or 0.0)
    stress = _clamp01(0.25 + t + 0.02 * intensity + 0.08 * context_flags["meeting_pressure"])
    suppression = _clamp01(0.15 + s + 0.01 * intensity + 0.10 * context_flags["privacy_on"])
    fatigue = _clamp01(0.20 + 0.15 * context_flags["quiet_mode"] + 0.08 * context_flags["daily_count"] + 0.10 * (1.0 - a))
    attention_drop = _clamp01(0.18 + 0.35 * fatigue + 0.20 * stress)
    valence = _clamp01(0.62 + v - 0.32 * stress - 0.10 * fatigue)
    arousal = _clamp01(0.20 + a + 0.18 * stress + 0.06 * intensity / 20.0)
    if _contains_any(support_text, CARE_MARKERS):
        stress = _clamp01(stress + 0.05)
    return {
        "valence": valence,
        "arousal": arousal,
        "stress": stress,
        "fatigue": fatigue,
        "attention_drop": attention_drop,
        "suppression": suppression,
    }


def _derive_observations(state: Dict[str, float], context_flags: Dict[str, float], support_text: str) -> Dict[str, float]:
    stress = state["stress"]
    fatigue = state["fatigue"]
    suppression = state["suppression"]
    attention_drop = state["attention_drop"]
    care_marker = 1.0 if _contains_any(support_text, CARE_MARKERS) else 0.0
    nudge_marker = 1.0 if _contains_any(support_text, NUDGE_MARKERS) else 0.0
    return {
        "face_presence_ratio": _clamp01(0.78 - 0.18 * attention_drop),
        "gaze_avert_ratio": _clamp01(0.20 + 0.35 * attention_drop + 0.08 * suppression),
        "head_motion_var": _clamp01(0.18 + 0.22 * fatigue + 0.18 * stress),
        "posture_slouch_score": _clamp01(0.22 + 0.30 * fatigue + 0.10 * stress),
        "fidget_score": _clamp01(0.18 + 0.35 * stress - 0.08 * context_flags["quiet_mode"]),
        "voice_energy": _clamp01(0.70 - 0.28 * fatigue - 0.18 * stress),
        "speech_rate": _clamp01(0.48 - 0.10 * fatigue + 0.10 * stress),
        "silence_ratio": _clamp01(0.18 + 0.20 * fatigue + 0.15 * context_flags["privacy_on"]),
        "prosody_stress": _clamp01(0.18 + 0.45 * stress + 0.08 * suppression),
        "attention_drop_proxy": _clamp01(attention_drop),
        "fatigue_proxy": _clamp01(fatigue),
        "stress_proxy": _clamp01(stress),
        "receptivity_proxy": _clamp01(0.60 - 0.22 * context_flags["busy_speaking"] - 0.18 * context_flags["privacy_on"] + 0.08 * care_marker + 0.04 * nudge_marker),
    }


def _derive_strategy(support_text: str, timing_label: str, context_flags: Dict[str, float]) -> str:
    if context_flags["privacy_on"] > 0.5:
        return "guard"
    if context_flags["cooldown_active"] > 0.5 and timing_label != "immediate":
        return "guard"
    if timing_label == "none":
        return "observe"
    if _contains_any(support_text, CARE_MARKERS):
        return "care"
    if _contains_any(support_text, NUDGE_MARKERS):
        return "nudge"
    return "nudge" if timing_label == "delay" else "care"


def _derive_timing(event: sqlite3.Row, support_text: str) -> str:
    if event["description"] == "event:TriggerFired" and support_text:
        return "immediate"
    intensity = int(event["intensity"] or 0)
    if intensity >= 5:
        return "delay"
    return "none"


def _find_support_text(events_by_ts: Dict[int, List[sqlite3.Row]], timestamp_ms: int) -> str:
    rows = events_by_ts.get(timestamp_ms, [])
    candidates = [str(row["description"] or "") for row in rows if str(row["description"] or "") != "event:TriggerFired"]
    if not candidates:
        return ""
    candidates.sort(key=len, reverse=True)
    return candidates[0]


def _reply_feedback(next_user_text: str | None, latency_ms: int | None) -> Dict[str, float]:
    if not next_user_text:
        return {
            "accepted": 0.0,
            "ignored": 1.0,
            "annoyed": 0.0,
            "response_latency_ms": float(10 * 60 * 1000),
            "helpfulness": 0.35,
            "acceptance": 0.15,
            "annoyance": 0.10,
        }
    annoyed = _contains_any(next_user_text, NEGATIVE_REPLY_MARKERS)
    accepted = not annoyed
    latency = float(latency_ms or 0)
    acceptance = _clamp01(0.75 if accepted else 0.10)
    annoyance = _clamp01(0.70 if annoyed else 0.12)
    helpfulness = _clamp01(0.62 if accepted else 0.18)
    return {
        "accepted": float(accepted),
        "ignored": 0.0,
        "annoyed": float(annoyed),
        "response_latency_ms": latency,
        "helpfulness": helpfulness,
        "acceptance": acceptance,
        "annoyance": annoyance,
    }


def _split_chronologically(episodes: Sequence[EpisodeWindowV1]) -> Dict[str, List[EpisodeWindowV1]]:
    items = sorted(episodes, key=lambda item: item.timestamp_range[0])
    total = len(items)
    train_end = int(total * 0.7)
    dev_end = train_end + int(total * 0.15)
    return {"train": items[:train_end], "dev": items[train_end:dev_end], "test": items[dev_end:]}


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_bridge_log_episodes(log_dir: str | Path, cfg: WeakLabelConfig | None = None) -> List[EpisodeWindowV1]:
    cfg = cfg or WeakLabelConfig()
    root = Path(log_dir)
    episodes: List[EpisodeWindowV1] = []
    files = sorted(root.glob(cfg.bridge_glob))
    time_cursor = 1_700_000_000_000
    for log_path in files:
        for line_index, line in enumerate(log_path.read_text(encoding="utf-8", errors="ignore").splitlines()):
            if "[CarePlanReady]" not in line:
                continue
            payload_text = line.split("[CarePlanReady]", 1)[1].strip()
            try:
                payload = ast.literal_eval(payload_text)
            except Exception:
                continue
            care_plan = payload.get("care_plan", {})
            reason = payload.get("reason", {})
            detail = payload.get("detail", {})
            v_sub = detail.get("V_sub", {})
            a_sub = detail.get("A_sub", {})
            t_sub = detail.get("T_sub", {})
            decision = str(care_plan.get("decision", "CARE")).upper()
            level = int(care_plan.get("level", 2) or 2)
            support_text = str(care_plan.get("text", ""))
            timestamp_ms = time_cursor + len(episodes) * 60_000 + line_index * 1000
            if decision == "CARE":
                timing_label = "immediate"
            elif decision == "DELAY":
                timing_label = "delay"
            else:
                timing_label = "none"
            tags = list(reason.get("tags", []) or t_sub.get("tags", []) or [])
            context_flags = {
                "busy_speaking": float(0.0),
                "privacy_on": float(0.0),
                "quiet_mode": float(0.0),
                "daily_count": float(1.0),
                "cooldown_active": float(1.0 if int(care_plan.get("cooldown_min", 0) or 0) > 0 else 0.0),
                "meeting_pressure": float(0.2),
                "social_load": float(0.3),
                "time_of_day": float(0.5),
                "recent_rejections": float(0.0),
                "recent_accepts": float(0.0),
            }
            state_labels = {
                "valence": _clamp01(0.65 - 0.40 * _safe_float(reason.get("S"), 0.0) - 0.20 * _safe_float(reason.get("T"), 0.0) + 0.10 * _safe_float(reason.get("V"), 0.0)),
                "arousal": _clamp01(0.15 + 0.25 * _safe_float(reason.get("A"), 0.0) + 0.12 * min(1.0, _safe_float(a_sub.get("rms"), 0.0) / 150.0)),
                "stress": _clamp01(0.25 + _safe_float(reason.get("T"), 0.0) + 0.10 * min(1.0, _safe_float(v_sub.get("expression_risk"), 0.0))),
                "fatigue": _clamp01(0.20 + 0.10 * min(1.0, _safe_float(a_sub.get("silence_sec"), 0.0) / 5.0)),
                "attention_drop": _clamp01(0.15 + 0.20 * min(1.0, _safe_float(a_sub.get("zcr"), 0.0))),
                "suppression": _clamp01(0.12 + _safe_float(reason.get("S"), 0.0)),
            }
            observations = {
                "face_presence_ratio": _clamp01(_safe_float(v_sub.get("face_ok"), 0.5)),
                "gaze_avert_ratio": _clamp01(0.20 + 0.20 * state_labels["attention_drop"]),
                "head_motion_var": _clamp01(0.20 + 0.15 * state_labels["fatigue"]),
                "posture_slouch_score": _clamp01(0.18 + 0.20 * state_labels["fatigue"]),
                "fidget_score": _clamp01(0.16 + 0.20 * state_labels["stress"]),
                "voice_energy": _clamp01(min(1.0, _safe_float(a_sub.get("rms"), 0.0) / 150.0)),
                "speech_rate": _clamp01(_safe_float(a_sub.get("zcr"), 0.0) * 4.0),
                "silence_ratio": _clamp01(min(1.0, _safe_float(a_sub.get("silence_sec"), 0.0) / 5.0)),
                "prosody_stress": _clamp01(state_labels["stress"]),
                "attention_drop_proxy": _clamp01(state_labels["attention_drop"]),
                "fatigue_proxy": _clamp01(state_labels["fatigue"]),
                "stress_proxy": _clamp01(state_labels["stress"]),
                "receptivity_proxy": _clamp01(0.55 - 0.10 * context_flags["cooldown_active"] + 0.08 * (1.0 if payload.get("delivery_mode") == "text" else 0.0)),
            }
            strategy_label = "care" if level >= 2 else "nudge"
            if decision not in {"CARE", "DELAY"}:
                strategy_label = "observe"
            if context_flags["cooldown_active"] > 0.5 and decision == "SKIP":
                strategy_label = "guard"
            persona = {
                "big5_openness": 0.5,
                "big5_conscientiousness": 0.5,
                "big5_extraversion": 0.45,
                "big5_agreeableness": 0.55,
                "big5_neuroticism": 0.5 + 0.1 * state_labels["stress"],
                "pref_directness": 0.55,
                "pref_checkin_frequency": 0.50,
                "pref_touchiness": 0.30,
            }
            episodes.append(
                EpisodeWindowV1(
                    episode_id=f"bridge_{log_path.stem}_{line_index}",
                    video_clip_path=f"bridge://{log_path.name}/{line_index}.mp4",
                    audio_clip_path=f"bridge://{log_path.name}/{line_index}.wav",
                    timestamp_range=(timestamp_ms - cfg.window_ms, timestamp_ms),
                    persona_profile=persona,
                    context_flags=context_flags,
                    state_labels=state_labels,
                    timing_label=timing_label,
                    strategy_label=strategy_label,
                    script_template_id=strategy_label,
                    feedback={
                        "accepted": 0.0,
                        "ignored": 1.0,
                        "annoyed": 0.0,
                        "response_latency_ms": float(300000),
                        "helpfulness": 0.40,
                        "acceptance": 0.20,
                        "annoyance": 0.10,
                    },
                    observations=observations,
                    extra={
                        "source": "bridge_log",
                        "log_file": log_path.name,
                        "delivery_mode": payload.get("delivery_mode", "text"),
                        "decision": decision,
                        "tags": tags,
                        "support_text": support_text,
                        "transcript": str(t_sub.get("transcript", "")),
                        "summary": str(t_sub.get("summary", "")),
                    },
                )
            )
    return episodes


def extract_weaklabel_episodes(db_path: str | Path, cfg: WeakLabelConfig | None = None) -> List[EpisodeWindowV1]:
    cfg = cfg or WeakLabelConfig()
    conn = sqlite3.connect(Path(db_path))
    conn.row_factory = sqlite3.Row
    events = _load_rows(
        conn,
        """
        select id, user_id, timestamp_ms, type, description, v, a, t, s, intensity, source
        from emotion_events
        order by user_id, timestamp_ms
        """,
    )
    messages = _load_rows(
        conn,
        """
        select id, user_id, sender, text, timestamp_ms, content_type
        from chat_messages
        order by user_id, timestamp_ms
        """,
    )
    conn.close()

    events_by_user: Dict[int, List[sqlite3.Row]] = defaultdict(list)
    messages_by_user: Dict[int, List[sqlite3.Row]] = defaultdict(list)
    for row in events:
        events_by_user[int(row["user_id"])].append(row)
    for row in messages:
        messages_by_user[int(row["user_id"])].append(row)

    episodes: List[EpisodeWindowV1] = []
    for user_id, user_events in events_by_user.items():
        user_messages = messages_by_user.get(user_id, [])
        persona = _derive_persona(user_events, user_messages)
        events_by_ts: Dict[int, List[sqlite3.Row]] = defaultdict(list)
        for row in user_events:
            events_by_ts[int(row["timestamp_ms"])].append(row)
        message_index = 0
        recent_accepts = 0
        recent_rejections = 0
        daily_counter: Counter[str] = Counter()
        for row in user_events:
            timestamp_ms = int(row["timestamp_ms"])
            date_key = str(timestamp_ms // (24 * 60 * 60 * 1000))
            daily_counter[date_key] += 1
            support_text = _find_support_text(events_by_ts, timestamp_ms)
            if row["description"] == "event:TriggerFired" and not support_text:
                continue
            while message_index < len(user_messages) and int(user_messages[message_index]["timestamp_ms"]) < timestamp_ms:
                message_index += 1
            next_user_text = None
            latency_ms = None
            for future in user_messages[message_index:]:
                future_ts = int(future["timestamp_ms"])
                if future_ts - timestamp_ms > cfg.reply_window_ms:
                    break
                if future["sender"] == "user":
                    next_user_text = str(future["text"] or "")
                    latency_ms = future_ts - timestamp_ms
                    break
            feedback = _reply_feedback(next_user_text, latency_ms)
            recent_accepts += int(feedback["accepted"] > 0.5)
            recent_rejections += int(feedback["annoyed"] > 0.5 or feedback["ignored"] > 0.5)
            recent_trigger = any(
                prev["description"] == "event:TriggerFired" and 0 < timestamp_ms - int(prev["timestamp_ms"]) <= 10 * 60 * 1000
                for prev in user_events
                if int(prev["timestamp_ms"]) < timestamp_ms
            )
            context_flags = _derive_context(
                timestamp_ms=timestamp_ms,
                daily_count=int(daily_counter[date_key]),
                cooldown_active=recent_trigger,
                recent_accepts=recent_accepts,
                recent_rejections=recent_rejections,
                message_text=support_text,
            )
            timing_label = _derive_timing(row, support_text)
            strategy_label = _derive_strategy(support_text, timing_label, context_flags)
            state_labels = _derive_state(row, support_text, context_flags)
            observations = _derive_observations(state_labels, context_flags, support_text)
            episodes.append(
                EpisodeWindowV1(
                    episode_id=f"weak_{user_id}_{timestamp_ms}_{int(row['id'])}",
                    video_clip_path=f"weaklabel://user{user_id}/{timestamp_ms}.mp4",
                    audio_clip_path=f"weaklabel://user{user_id}/{timestamp_ms}.wav",
                    timestamp_range=(timestamp_ms - cfg.window_ms, timestamp_ms),
                    persona_profile=persona,
                    context_flags=context_flags,
                    state_labels=state_labels,
                    timing_label=timing_label,
                    strategy_label=strategy_label,
                    script_template_id=strategy_label if support_text else "observe_only",
                    feedback=feedback,
                    observations=observations,
                    extra={
                        "source": "weaklabel",
                        "user_id": int(user_id),
                        "event_type": str(row["type"] or ""),
                        "event_description": str(row["description"] or ""),
                        "support_text": support_text,
                    },
                )
            )
    return episodes


def write_weaklabel_dataset(
    db_path: str | Path,
    output_root: str | Path,
    cfg: WeakLabelConfig | None = None,
) -> Dict[str, Any]:
    cfg = cfg or WeakLabelConfig()
    episodes = extract_weaklabel_episodes(db_path, cfg)
    bridge_root = Path(db_path).resolve().parent.parent / "logs"
    if cfg.include_bridge and bridge_root.exists():
        episodes.extend(extract_bridge_log_episodes(bridge_root, cfg))
    splits = _split_chronologically(episodes)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    for split_name, split_items in splits.items():
        save_episodes(root / f"{split_name}.jsonl", split_items)
    manifest = {
        "db_path": str(db_path),
        "bridge_root": str(bridge_root),
        "include_bridge": bool(cfg.include_bridge),
        "counts": {name: len(items) for name, items in splits.items()},
        "label_distribution": {
            "timing": Counter(item.timing_label for item in episodes),
            "strategy": Counter(item.strategy_label for item in episodes),
        },
    }
    save_json(root / "manifest.json", manifest)
    return manifest
