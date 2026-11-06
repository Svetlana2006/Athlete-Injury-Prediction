/**
 * ATHLETEIQ – API Service Layer
 * All backend calls go through this module so the base URL is configured once.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ─────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────

export interface DashboardSummary {
  total_athletes: number;
  low_risk: number;
  moderate_risk: number;
  elevated_risk: number;
  high_risk: number;
  data_source: string;
}

export interface AthleteAtRisk {
  id: string;
  name: string;
  sport: string;
  position: string;
  team: string;
  risk_prob: number;
  risk_category: string;
  acwr: number;
}

export interface ModelInfo {
  model_version: string;
  roc_auc: number;
  pr_auc: number;
  seq_len: number;
  input_dim: number;
  loss: string;
  device: string;
}

export interface Athlete {
  id: string;
  name: string;
  sport: string;
  position: string;
  age: number;
  team: string;
  scenario: string;
  height_cm: number;
  weight_kg: number;
  dominant_foot: string;
  latest_risk_prob?: number;
  risk_category?: string;
}

export interface DailyRecord {
  date: string;
  athlete_id: string;
  rpe: number;
  duration_minutes: number;
  session_rpe: number;
  daily_load: number;
  weekly_load: number;
  fatigue: number;
  soreness: number;
  mood: number;
  readiness: number;
  sleep_duration: number;
  sleep_quality: number;
  stress: number;
  total_distance_m: number;
  high_speed_distance_m: number;
  sprint_distance_m: number;
  max_speed_m_s: number;
  mean_speed_m_s: number;
  prior_injury_count: number;
  has_prior_injury: number;
  is_rest_day: number;
  has_gps: number;
  illness_last_7d: number;
}

export interface AthleteHistory {
  athlete_id: string;
  athlete_name: string;
  daily_records: DailyRecord[];
  prediction_history: PredictionRecord[];
}

export interface PredictionRecord {
  athlete_id: string;
  predicted_probability: number;
  risk_category: string;
  attention_weights: number[];
  model_version: string;
  prediction_source: string;
  timestamp: string;
}

export interface BenchmarkResult {
  model: string;
  loss: string;
  roc_auc: number;
  pr_auc: number;
  brier: number;
  top10_recall: number;
  top10_lift: number | null;
  params: number;
  is_production: boolean;
  note: string;
}

export interface ModelConfig {
  architecture: string;
  description: string;
  input_window: number;
  input_features: number;
  hidden_dim: number;
  num_heads: number;
  num_layers: number;
  dim_feedforward: number;
  dropout: number;
  loss_function: string;
  focal_alpha: number;
  focal_gamma: number;
  optimizer: string;
  lr: number;
  lr_schedule?: string;
  batch_size: number;
  max_epochs: number;
  best_epoch: number;
  total_params: number;
  run_id: string;
}

export interface FeatureInfo {
  name: string;
  group: string;
  description: string;
}

// ─────────────────────────────────────────────────────────
// Fetch helper
// ─────────────────────────────────────────────────────────

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, { cache: "no-store", ...init });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status} at ${path}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ─────────────────────────────────────────────────────────
// Dashboard
// ─────────────────────────────────────────────────────────

export const getDashboardSummary  = () => apiFetch<DashboardSummary>("/api/v1/dashboard/summary");
export const getAthletesAtRisk    = () => apiFetch<AthleteAtRisk[]>("/api/v1/dashboard/athletes-at-risk");
export const getModelInfo         = () => apiFetch<ModelInfo>("/api/v1/dashboard/model-info");

// ─────────────────────────────────────────────────────────
// Athletes
// ─────────────────────────────────────────────────────────

export const listAthletes         = () => apiFetch<Athlete[]>("/api/v1/athletes");
export const getAthlete           = (id: string) => apiFetch<Athlete>(`/api/v1/athletes/${id}`);
export const getAthleteHistory    = (id: string) => apiFetch<AthleteHistory>(`/api/v1/athletes/${id}/history`);

// ─────────────────────────────────────────────────────────
// Predictions
// ─────────────────────────────────────────────────────────

/** RAW_FEATURE_COLUMNS order – must match backend exactly */
const FEATURES = [
  "rpe","duration_minutes","session_rpe","daily_load","weekly_load",
  "fatigue","soreness","mood","readiness",
  "sleep_duration","sleep_quality","stress",
  "total_distance_m","high_speed_distance_m","sprint_distance_m",
  "max_speed_m_s","mean_speed_m_s",
  "prior_injury_count","has_prior_injury","is_rest_day","has_gps","illness_last_7d",
];

/** Convert the last 14 daily records into a [14][22] sequence for the model. */
export function buildSequence(records: DailyRecord[]): number[][] {
  const last14 = records.slice(-14);
  return last14.map(r =>
    FEATURES.map(f => {
      const rec = r as unknown as Record<string, number>;
      const v = rec[f];
      return typeof v === "number" ? v : 0;
    })
  );
}

export async function runPrediction(
  athleteId: string,
  records: DailyRecord[]
): Promise<PredictionRecord> {
  const sequence = buildSequence(records);
  return apiFetch<PredictionRecord>("/api/v1/predictions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ athlete_id: athleteId, sequence }),
  });
}

export const getPredictionHistory = (id: string) =>
  apiFetch<PredictionRecord[]>(`/api/v1/predictions/${id}`);

export const resetPredictions = () =>
  apiFetch<{ status: string; message: string }>("/api/v1/predictions/reset", {
    method: "DELETE",
  });

// ─────────────────────────────────────────────────────────
// Model Insights
// ─────────────────────────────────────────────────────────

export const getBenchmarkResults  = () => apiFetch<BenchmarkResult[]>("/api/v1/model-insights/results");
export const getBestModelConfig   = () => apiFetch<ModelConfig>("/api/v1/model-insights/config");
export const getFeatureList       = () => apiFetch<FeatureInfo[]>("/api/v1/model-insights/features");
