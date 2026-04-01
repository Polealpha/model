import { ActiveCareFeedbackLabel, DeviceStatus, EngineMode, RiskDetail, RiskScores } from "../types";
import { apiPost } from "./apiClient";
import type { PersonalityProfile } from "./activationService";

export interface ResearchPersonaProfile {
  persona_id: string;
  big_five: number[];
  interaction_preferences: Record<string, number>;
  notes?: Record<string, unknown>;
}

export interface ResearchContextFlags {
  busy_speaking: boolean;
  privacy_on: boolean;
  quiet_mode: boolean;
  daily_count: number;
  cooldown_active: boolean;
  scene: string;
}

export interface ResearchEpisodeWindowPayload {
  sample_id: string;
  video_clip_path: string;
  audio_clip_path: string;
  timestamp_range: [number, number];
  persona_profile: ResearchPersonaProfile;
  context_flags: ResearchContextFlags;
  feature_vector: Record<string, number>;
  state_labels: {
    valence: number;
    arousal: number;
    stress: number;
    fatigue: number;
    attention_drop: number;
    suppression: number;
  };
  source: string;
}

export interface ResearchFeedbackPayload {
  message_id: string;
  sample_id?: string | null;
  accepted: boolean;
  ignored: boolean;
  annoyed: boolean;
  response_latency_ms: number;
  source: string;
}

export interface ResearchDecisionPayload {
  timing_decision?: {
    decision: string;
    score: number;
    uncertainty: number;
    why_codes?: string[];
  };
  strategy_plan?: {
    strategy_level: string;
    outline_steps?: string[];
    utterance_constraints?: unknown;
  };
  care_utterance?: {
    draft_text: string;
    confirmation_question?: string;
    repair_text?: string;
  };
  sample_id?: string;
}

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));

const keywordScore = (items: string[], keywords: string[]) => {
  const text = items.join(" ").toLowerCase();
  return keywords.some((keyword) => text.includes(keyword)) ? 1 : 0;
};

const scaleUnit = (value: unknown, fallback = 0, divisor = 1) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return clamp01(numeric / divisor);
};

export const personalityToPersonaProfile = (
  profile: PersonalityProfile | null | undefined
): ResearchPersonaProfile => {
  const traits = Array.isArray(profile?.traits) ? profile.traits : [];
  const boundaries = Array.isArray(profile?.boundaries) ? profile.boundaries : [];
  const signals = Array.isArray(profile?.signals) ? profile.signals : [];
  const responseStyle = String(profile?.response_style || "");
  const careStyle = String(profile?.care_style || "");
  const summary = String(profile?.summary || "");
  const decisionText = `${responseStyle} ${careStyle} ${summary} ${traits.join(" ")} ${boundaries.join(" ")} ${signals.join(" ")}`;

  const extraversion = clamp01(
    0.45 +
      0.20 * keywordScore(traits, ["外向", "健谈", "社交", "主动"]) -
      0.20 * keywordScore(traits, ["内向", "慢热", "独处"]) +
      0.10 * keywordScore([responseStyle], ["直接", "外放"])
  );
  const conscientiousness = clamp01(
    0.50 +
      0.18 * keywordScore([decisionText], ["有条理", "计划", "稳定", "规律"]) -
      0.10 * keywordScore([decisionText], ["随性", "灵活", "即兴"])
  );
  const agreeableness = clamp01(
    0.52 +
      0.15 * keywordScore([decisionText], ["温和", "合作", "体贴", "包容"]) -
      0.10 * keywordScore([decisionText], ["防御", "强硬"])
  );
  const neuroticism = clamp01(
    0.40 +
      0.22 * keywordScore(signals, ["焦虑", "敏感", "紧张", "压力"]) +
      0.12 * keywordScore([summary], ["情绪波动", "容易焦虑"])
  );
  const openness = clamp01(
    0.48 +
      0.14 * keywordScore([decisionText], ["开放", "好奇", "探索"]) -
      0.06 * keywordScore([decisionText], ["保守"])
  );

  const prefersExternalCare = clamp01(
    0.35 +
      0.30 * keywordScore([decisionText], ["希望陪伴", "需要安慰", "喜欢被关注", "倾诉"]) -
      0.25 * keywordScore(boundaries, ["不想被打扰", "自己消化", "不喜欢主动打断"])
  );
  const prefersDirectness = clamp01(
    0.45 +
      0.25 * keywordScore([responseStyle], ["直接", "明确"]) -
      0.20 * keywordScore([responseStyle], ["委婉", "缓和"])
  );
  const prefersHigherFrequency = clamp01(
    0.30 +
      0.25 * keywordScore([careStyle], ["主动", "及时", "频繁"]) -
      0.20 * keywordScore(boundaries, ["低频", "少打扰"])
  );

  return {
    persona_id: "desktop-live-persona",
    big_five: [openness, conscientiousness, extraversion, agreeableness, neuroticism],
    interaction_preferences: {
      prefers_external_care: prefersExternalCare,
      prefers_directness: prefersDirectness,
      prefers_higher_frequency: prefersHigherFrequency,
    },
    notes: {
      confidence: Number(profile?.confidence || 0),
      sample_count: Number(profile?.sample_count || 0),
      summary: profile?.summary || "",
    },
  };
};

export const buildResearchEpisodeWindow = (args: {
  scores: RiskScores;
  riskDetail: RiskDetail | null;
  deviceStatus: DeviceStatus | null;
  mode: EngineMode;
  personality: PersonalityProfile | null | undefined;
  nowMs?: number;
}): ResearchEpisodeWindowPayload => {
  const nowMs = args.nowMs ?? Date.now();
  const startMs = nowMs - 60_000;
  const vSub = (args.riskDetail?.V_sub || {}) as Record<string, unknown>;
  const aSub = (args.riskDetail?.A_sub || {}) as Record<string, unknown>;
  const tSub = (args.riskDetail?.T_sub || {}) as Record<string, unknown>;
  const silenceSec = Number(aSub.silence_sec ?? 0);
  const rms = Number(aSub.rms ?? 0);
  const fatigue = scaleUnit(vSub.fatigue ?? args.scores.V, scaleUnit(args.scores.V), 1);
  const attentionDrop = scaleUnit(vSub.attention_drop ?? args.scores.T, scaleUnit(args.scores.T), 1);
  const stress = clamp01(Number.isFinite(Number(args.scores.S)) ? Number(args.scores.S) : 0);
  const arousal = clamp01(Number.isFinite(Number(args.scores.A)) ? Number(args.scores.A) : stress);
  const suppression = clamp01(0.28 + 0.35 * scaleUnit(vSub.expression_valid, 0) * (1 - scaleUnit(vSub.expression_confidence, 0)));
  const valence = clamp01(0.68 - 0.40 * stress - 0.22 * fatigue - 0.16 * suppression);
  const busySpeaking = silenceSec < 0.8 && rms > 0.01;
  const scene = args.deviceStatus?.ssid ? "desk" : "unknown";

  return {
    sample_id: `desktop-${nowMs}`,
    video_clip_path: "desktop://live/video",
    audio_clip_path: "desktop://live/audio",
    timestamp_range: [startMs, nowMs],
    persona_profile: personalityToPersonaProfile(args.personality),
    context_flags: {
      busy_speaking: busySpeaking,
      privacy_on: args.mode === "privacy",
      quiet_mode: args.mode === "dnd",
      daily_count: 0,
      cooldown_active: false,
      scene,
    },
    feature_vector: {
      face_presence_ratio: scaleUnit(vSub.frame_decode_ok ?? vSub.expression_valid, 0),
      gaze_avert_ratio: scaleUnit(vSub.gaze_avert_ratio, 0.18),
      head_motion_var: scaleUnit(tSub.head_motion_var ?? tSub.motion_var, 0.16),
      posture_slouch_score: scaleUnit(tSub.posture_slouch_score ?? fatigue, fatigue),
      fidget_score: scaleUnit(tSub.fidget_score ?? tSub.motion_energy, 0.12),
      voice_energy: scaleUnit(rms, 0, 0.08),
      speech_rate: scaleUnit(aSub.speech_rate, 0.45, 6),
      silence_ratio: scaleUnit(silenceSec, 0, 6),
      prosody_stress: scaleUnit(aSub.voice_tension ?? stress, stress),
      attention_drop_proxy: attentionDrop,
      fatigue_proxy: fatigue,
      stress_proxy: stress,
      receptivity_proxy: clamp01(0.60 - 0.30 * Number(busySpeaking) - 0.20 * Number(args.mode === "privacy") - 0.10 * Number(args.mode === "dnd")),
      expression_class_id: scaleUnit(vSub.expression_class_id, 0, 8),
      expression_confidence: scaleUnit(vSub.expression_confidence, 0),
    },
    state_labels: {
      valence,
      arousal,
      stress,
      fatigue,
      attention_drop: attentionDrop,
      suppression,
    },
    source: "desktop_live",
  };
};

export const buildResearchFeedbackPayload = (args: {
  messageId: string;
  sampleId?: string | null;
  feedback: ActiveCareFeedbackLabel;
  responseLatencyMs: number;
  source?: string;
}): ResearchFeedbackPayload => ({
  message_id: args.messageId,
  sample_id: args.sampleId ?? null,
  accepted: args.feedback === "accepted",
  ignored: args.feedback === "ignored",
  annoyed: args.feedback === "annoyed",
  response_latency_ms: Math.max(0, Math.floor(args.responseLatencyMs)),
  source: args.source || "desktop_active_care",
});

export const submitResearchInference = async (payload: ResearchEpisodeWindowPayload): Promise<ResearchDecisionPayload> => {
  return apiPost("/api/research/infer", payload, true);
};

export const submitResearchFeedback = async (payload: ResearchFeedbackPayload) => {
  return apiPost("/api/research/feedback", payload, true);
};
