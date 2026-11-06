"""
ATHLETEIQ – Model Insights Route
==================================
GET /api/v1/model-insights/results      → benchmark results
GET /api/v1/model-insights/config       → best-model training configuration
GET /api/v1/model-insights/feature-importance  → RAW_FEATURE_COLUMNS list
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/model-insights", tags=["model-insights"])

# ---------------------------------------------------------------------------
# Hard-coded from actual experiment artifacts (nn_benchmark_results.json
# and artifacts/runs/20261104_153047_Transformer/results.json)
# ---------------------------------------------------------------------------

BENCHMARK_RESULTS = [
    {
        "model":       "Transformer + Focal Loss",
        "loss":        "FocalLoss(α=0.25, γ=2.0)",
        "roc_auc":     0.9191,
        "pr_auc":      0.0226,
        "brier":       0.0120,
        "top10_recall": 0.8033,
        "top10_lift":  8.03,
        "params":      72642,
        "is_production": True,
        "note":        "Highest ROC-AUC – selected for production inference",
    },
    {
        "model":       "Transformer + BCE",
        "loss":        "BCEWithLogitsLoss(pos_weight=5)",
        "roc_auc":     0.9011,
        "pr_auc":      0.0170,
        "brier":       0.0333,
        "top10_recall": 0.6885,
        "top10_lift":  None,
        "params":      72642,
        "is_production": False,
        "note":        "Balanced baseline – susceptible to class noise",
    },
    {
        "model":       "GRU",
        "loss":        "BCEWithLogitsLoss",
        "roc_auc":     0.8912,
        "pr_auc":      0.0206,
        "brier":       0.0252,
        "top10_recall": 0.6557,
        "top10_lift":  None,
        "params":      46082,
        "is_production": False,
        "note":        "Recurrent baseline – sequential memory degrades >20 days",
    },
    {
        "model":       "CNN-GRU",
        "loss":        "BCEWithLogitsLoss",
        "roc_auc":     0.8196,
        "pr_auc":      0.0401,
        "brier":       0.0298,
        "top10_recall": 0.5574,
        "top10_lift":  None,
        "params":      58562,
        "is_production": False,
        "note":        "Highest PR-AUC – best precision on rare events under extreme thresholds",
    },
]

BEST_MODEL_CONFIG = {
    "architecture":    "TransformerEncoder",
    "description":     "Multi-Head Self-Attention Temporal Encoder",
    "input_window":    14,
    "input_features":  22,
    "hidden_dim":      64,
    "num_heads":       4,
    "num_layers":      2,
    "dim_feedforward": 128,
    "dropout":         0.25,
    "loss_function":   "FocalLoss",
    "focal_alpha":     0.25,
    "focal_gamma":     2.0,
    "optimizer":       "AdamW",
    "lr":              5e-5,
    "lr_min":          1e-6,
    "warmup_epochs":   2,
    "lr_schedule":     "LinearWarmup → CosineAnnealing",
    "batch_size":      128,
    "max_epochs":      50,
    "best_epoch":      13,
    "patience":        10,
    "weight_decay":    0.01,
    "max_grad_norm":   1.0,
    "train_cutoff":    "2021-01-01",
    "split_strategy":  "Temporal (strict time-based)",
    "total_params":    72642,
    "run_id":          "20261104_153047_Transformer",
}

RAW_FEATURE_COLUMNS = [
    {"name": "rpe",                   "group": "Workload", "description": "Rate of perceived exertion (1–10)"},
    {"name": "duration_minutes",      "group": "Workload", "description": "Session duration in minutes"},
    {"name": "session_rpe",           "group": "Workload", "description": "Session RPE × Duration (AU)"},
    {"name": "daily_load",            "group": "Workload", "description": "Daily training load (AU)"},
    {"name": "weekly_load",           "group": "Workload", "description": "7-day cumulative load (AU)"},
    {"name": "fatigue",               "group": "Wellness", "description": "Subjective fatigue score (1–10)"},
    {"name": "soreness",              "group": "Wellness", "description": "Muscle soreness rating (1–10)"},
    {"name": "mood",                  "group": "Wellness", "description": "Mood rating (1–10)"},
    {"name": "readiness",             "group": "Wellness", "description": "Perceived readiness (1–10)"},
    {"name": "sleep_duration",        "group": "Wellness", "description": "Sleep duration in hours"},
    {"name": "sleep_quality",         "group": "Wellness", "description": "Subjective sleep quality (1–10)"},
    {"name": "stress",                "group": "Wellness", "description": "Perceived stress level (1–10)"},
    {"name": "total_distance_m",      "group": "GPS",      "description": "Total GPS distance (metres)"},
    {"name": "high_speed_distance_m", "group": "GPS",      "description": "High-speed running distance (metres)"},
    {"name": "sprint_distance_m",     "group": "GPS",      "description": "Sprint distance (metres)"},
    {"name": "max_speed_m_s",         "group": "GPS",      "description": "Max speed (m/s)"},
    {"name": "mean_speed_m_s",        "group": "GPS",      "description": "Mean session speed (m/s)"},
    {"name": "prior_injury_count",    "group": "Clinical", "description": "Total historical injuries"},
    {"name": "has_prior_injury",      "group": "Clinical", "description": "Binary: any prior injury"},
    {"name": "is_rest_day",           "group": "Clinical", "description": "Binary: scheduled rest day"},
    {"name": "has_gps",               "group": "GPS",      "description": "Binary: GPS data available"},
    {"name": "illness_last_7d",       "group": "Clinical", "description": "Binary: illness in last 7 days"},
]


@router.get("/results")
def benchmark_results() -> list[dict]:
    return BENCHMARK_RESULTS


@router.get("/config")
def best_model_config() -> dict:
    return BEST_MODEL_CONFIG


@router.get("/features")
def feature_list() -> list[dict]:
    return RAW_FEATURE_COLUMNS
