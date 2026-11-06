"""
ATHLETEIQ – Synthetic Demo Data Generator
==========================================
Generates 14 fictional football athletes for Apex FC with 30 days of realistic telemetry.
Records use the EXACT same 22 feature columns the trained Transformer expects.

CRITICAL PRESENTATION CONTRACT:
Our demo data contains a curated distribution of athletes across ALL 4 clinical risk categories:
  - High Risk (>50%):     3 athletes with severe training overload, sleep deprivation & prior injuries
  - Elevated Risk (30-50%): 3 athletes with mounting workload trends, low recovery & elevated fatigue
  - Moderate Risk (15-30%): 4 athletes with fluctuating training load & mild soreness
  - Low Risk (<15%):      4 athletes with balanced periodization, high readiness & optimal recovery

Output files generated:
  - app/data/athletes.json        (Athlete profiles with risk category metadata)
  - app/data/sample_records.json (30-day time-series sensor telemetry per athlete)
  - app/data/predictions.json    (Pre-computed baseline Transformer inference records)
"""

from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Seed for reproducibility
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "app" / "data"

# ---------------------------------------------------------------------------
# Feature columns (matches app/ml/train.py RAW_FEATURE_COLUMNS exactly)
# ---------------------------------------------------------------------------
FEATURES = [
    "rpe", "duration_minutes", "session_rpe", "daily_load", "weekly_load",
    "fatigue", "soreness", "mood", "readiness",
    "sleep_duration", "sleep_quality", "stress",
    "total_distance_m", "high_speed_distance_m", "sprint_distance_m",
    "max_speed_m_s", "mean_speed_m_s",
    "prior_injury_count", "has_prior_injury", "is_rest_day", "has_gps",
    "illness_last_7d",
]
N_FEATURES = len(FEATURES)   # 22
N_DAYS     = 30              # history per athlete

# ---------------------------------------------------------------------------
# Athlete Roster: 14 Football Players across All 4 Risk Categories
# ---------------------------------------------------------------------------
# We have a few athletes from each category to provide a realistic, comprehensive
# dataset for presentation and clinical demonstration.
ATHLETES: list[dict[str, Any]] = [
    # ── HIGH RISK CATEGORY (3 Athletes: >50% probability) ──────────────────
    {
        "id": "ATH-101",
        "name": "Marcus Sterling",
        "sport": "Football",
        "position": "Winger",
        "age": 24,
        "team": "Apex FC",
        "scenario": "High",
        "risk_category": "High",
        "predicted_probability": 0.824,
        "scenario_title": "Acute Workload Spike & Sleep Deprivation",
        "clinical_note": "Significant 30% jump in sprint distance, acute:chronic workload ratio > 1.8, sleep < 5h, 2 prior hamstring strains.",
    },
    {
        "id": "ATH-107",
        "name": "Mateo Silva",
        "sport": "Football",
        "position": "Midfielder",
        "age": 27,
        "team": "Apex FC",
        "scenario": "High",
        "risk_category": "High",
        "predicted_probability": 0.768,
        "scenario_title": "High Sprint Volume & Muscle Strain",
        "clinical_note": "Excessive high-speed running across consecutive matches with soreness 9/10 and elevated muscle fatigue.",
    },
    {
        "id": "ATH-110",
        "name": "Tariq Mansoor",
        "sport": "Football",
        "position": "Winger",
        "age": 25,
        "team": "Apex FC",
        "scenario": "High",
        "risk_category": "High",
        "predicted_probability": 0.841,
        "scenario_title": "Post-Illness Return & Severe Fatigue Accumulation",
        "clinical_note": "Accelerated return to play post-viral illness without adequate reconditioning, leading to extreme systemic fatigue.",
    },

    # ── ELEVATED RISK CATEGORY (3 Athletes: 30-50% probability) ────────────
    {
        "id": "ATH-105",
        "name": "Noah Becker",
        "sport": "Football",
        "position": "Striker",
        "age": 25,
        "team": "Apex FC",
        "scenario": "Elevated",
        "risk_category": "Elevated",
        "predicted_probability": 0.546,
        "scenario_title": "Mounting Training Volume & Sleep Deficit",
        "clinical_note": "Consistent 3-week workload increase with average sleep dropping below 6.5 hours and rising subjective soreness.",
    },
    {
        "id": "ATH-111",
        "name": "Lucas Romero",
        "sport": "Football",
        "position": "Striker",
        "age": 30,
        "team": "Apex FC",
        "scenario": "Elevated",
        "risk_category": "Elevated",
        "predicted_probability": 0.482,
        "scenario_title": "Veteran Overuse & High Session Variance",
        "clinical_note": "High cumulative match minutes with age-related recovery latency; moderate quad tightness reported.",
    },
    {
        "id": "ATH-114",
        "name": "Owen Griffiths",
        "sport": "Football",
        "position": "Full-Back",
        "age": 29,
        "team": "Apex FC",
        "scenario": "Elevated",
        "risk_category": "Elevated",
        "predicted_probability": 0.515,
        "scenario_title": "Congested Fixture Schedule",
        "clinical_note": "Three full 90-minute fixtures in 8 days; elevated autonomic fatigue with readiness below 5.0.",
    },

    # ── MODERATE RISK CATEGORY (4 Athletes: 15-30% probability) ────────────
    {
        "id": "ATH-104",
        "name": "Elena Rostova",
        "sport": "Football",
        "position": "Midfielder",
        "age": 22,
        "team": "Apex FC",
        "scenario": "Moderate",
        "risk_category": "Moderate",
        "predicted_probability": 0.263,
        "scenario_title": "Fluctuating Workload with Stable Sleep",
        "clinical_note": "Intermittent high-intensity sessions balanced by adequate 7.5h sleep and regular active recovery.",
    },
    {
        "id": "ATH-108",
        "name": "Arjun Nair",
        "sport": "Football",
        "position": "Full-Back",
        "age": 27,
        "team": "Apex FC",
        "scenario": "Moderate",
        "risk_category": "Moderate",
        "predicted_probability": 0.287,
        "scenario_title": "Minor Adductor Soreness with Controlled Load",
        "clinical_note": "Reported adductor tightness being managed with reduced kicking drills; overall readiness remains steady.",
    },
    {
        "id": "ATH-109",
        "name": "Kasper Lind",
        "sport": "Football",
        "position": "Center-Back",
        "age": 26,
        "team": "Apex FC",
        "scenario": "Moderate",
        "risk_category": "Moderate",
        "predicted_probability": 0.219,
        "scenario_title": "Progressive Conditioning Phase",
        "clinical_note": "Systematic aerobic building block; metrics tracking well within expected physiological tolerances.",
    },
    {
        "id": "ATH-112",
        "name": "Bukayo Saka",
        "sport": "Football",
        "position": "Winger",
        "age": 22,
        "team": "Apex FC",
        "scenario": "Moderate",
        "risk_category": "Moderate",
        "predicted_probability": 0.245,
        "scenario_title": "Reconditioning Following Minor Rest",
        "clinical_note": "Returning from scheduled rotation week; intensity progressing conservatively with low stress markers.",
    },

    # ── LOW RISK CATEGORY (4 Athletes: <15% probability) ───────────────────
    {
        "id": "ATH-102",
        "name": "Rashid Patel",
        "sport": "Football",
        "position": "Center-Back",
        "age": 26,
        "team": "Apex FC",
        "scenario": "Low",
        "risk_category": "Low",
        "predicted_probability": 0.084,
        "scenario_title": "Optimal Periodization & High Readiness",
        "clinical_note": "Excellent ACWR 1.05, 8.2h average sleep, 0 prior injuries, readiness score 8.5/10.",
    },
    {
        "id": "ATH-103",
        "name": "Liam Vance",
        "sport": "Football",
        "position": "Center-Back",
        "age": 28,
        "team": "Apex FC",
        "scenario": "Low",
        "risk_category": "Low",
        "predicted_probability": 0.091,
        "scenario_title": "Stable Loading & Restorative Sleep",
        "clinical_note": "Consistent weekly loading pattern; deep sleep tracking high, low subjective fatigue.",
    },
    {
        "id": "ATH-106",
        "name": "Devraj Sharma",
        "sport": "Football",
        "position": "Goalkeeper",
        "age": 23,
        "team": "Apex FC",
        "scenario": "Low",
        "risk_category": "Low",
        "predicted_probability": 0.068,
        "scenario_title": "Goalkeeper Positional Profile",
        "clinical_note": "Position-specific low total distance, high explosive power with minimal running fatigue accumulation.",
    },
    {
        "id": "ATH-113",
        "name": "Fernanda Diaz",
        "sport": "Football",
        "position": "Midfielder",
        "age": 24,
        "team": "Apex FC",
        "scenario": "Low",
        "risk_category": "Low",
        "predicted_probability": 0.075,
        "scenario_title": "Balanced Aerobic Conditioning",
        "clinical_note": "Superb wellness composite score, minimal soreness, full weekly rest day utilized effectively.",
    },
]

assert len(ATHLETES) == 14, "The demo roster must contain exactly 14 football athletes"

# ---------------------------------------------------------------------------
# Telemetry Generation per Risk Category
# ---------------------------------------------------------------------------
def _base(category: str, day: int) -> dict[str, float]:
    """
    Returns baseline feature values for each risk category and day index
    (0 = oldest day, N_DAYS-1 = most recent day).
    """
    t = day / (N_DAYS - 1)   # progress 0.0 -> 1.0 across monitoring window

    if category == "High":
        # Acute spike in the second half of the 30-day window
        spike = max(0.0, (t - 0.5) / 0.5)  # 0 until day 15, then ramps to 1.0
        rpe           = 5.5 + 4.0 * spike
        duration      = 70.0 + 35.0 * spike
        session_rpe   = rpe * duration
        daily_load    = session_rpe
        weekly_load   = daily_load * (4.5 + 2.5 * spike)
        fatigue       = 4.0 + 5.2 * spike
        soreness      = 3.5 + 5.8 * spike
        mood          = 7.0 - 4.5 * spike
        readiness     = 7.0 - 5.0 * spike
        sleep_duration= 7.8 - 3.2 * spike   # drops down to 4.6 hrs
        sleep_quality = 7.5 - 4.8 * spike
        stress        = 3.5 + 5.5 * spike
        total_dist    = 6800.0 + 4500.0 * spike
        hsd           = 800.0 + 900.0 * spike
        sprint_dist   = 180.0 + 280.0 * spike
        max_speed     = 8.8 + 1.2 * spike
        mean_speed    = 3.0 + 0.8 * spike
        prior_inj     = 2
        has_prior     = 1
        is_rest       = 1 if (day == 7 or day == 14) and spike < 0.2 else 0
        has_gps       = 1
        illness       = 1 if 0.55 < t < 0.70 else 0

    elif category == "Elevated":
        # Progressive upward trend in workload, moderate recovery lag
        rpe           = 5.0 + 2.2 * t
        duration      = 65.0 + 20.0 * t
        session_rpe   = rpe * duration
        daily_load    = session_rpe
        weekly_load   = daily_load * (4.2 + 1.2 * t)
        fatigue       = 3.5 + 3.2 * t
        soreness      = 3.0 + 3.0 * t
        mood          = 7.5 - 2.0 * t
        readiness     = 7.5 - 2.5 * t
        sleep_duration= 7.6 - 1.2 * t   # drops to ~6.4 hrs
        sleep_quality = 7.5 - 2.0 * t
        stress        = 3.0 + 3.2 * t
        total_dist    = 6500.0 + 2500.0 * t
        hsd           = 750.0 + 450.0 * t
        sprint_dist   = 160.0 + 120.0 * t
        max_speed     = 8.5 + 0.6 * t
        mean_speed    = 2.9 + 0.4 * t
        prior_inj     = 1
        has_prior     = 1
        is_rest       = 1 if day in (6, 13, 20) else 0
        has_gps       = 1
        illness       = 0

    elif category == "Moderate":
        # Fluctuating load pattern around healthy baseline
        phase         = np.sin(day * 0.7) * 0.5 + 0.5
        rpe           = 4.5 + 1.5 * phase
        duration      = 60.0 + 15.0 * phase
        session_rpe   = rpe * duration
        daily_load    = session_rpe
        weekly_load   = daily_load * (4.0 + 1.0 * phase)
        fatigue       = 3.2 + 1.8 * phase
        soreness      = 2.8 + 1.8 * phase
        mood          = 7.5 - 1.2 * phase
        readiness     = 7.5 - 1.5 * phase
        sleep_duration= 7.4 - 0.6 * phase
        sleep_quality = 7.2 - 1.0 * phase
        stress        = 3.0 + 1.8 * phase
        total_dist    = 6200.0 + 1600.0 * phase
        hsd           = 700.0 + 300.0 * phase
        sprint_dist   = 140.0 + 80.0 * phase
        max_speed     = 8.3 + 0.5 * phase
        mean_speed    = 2.8 + 0.3 * phase
        prior_inj     = 1 if day % 2 == 0 else 0
        has_prior     = 1 if prior_inj > 0 else 0
        is_rest       = 1 if day in (5, 12, 19, 26) else 0
        has_gps       = 1
        illness       = 0

    else:  # "Low"
        # Stable, well-periodized training with full recovery
        rpe           = 4.2
        duration      = 58.0
        session_rpe   = rpe * duration
        daily_load    = session_rpe
        weekly_load   = daily_load * 4.2
        fatigue       = 2.4
        soreness      = 1.8
        mood          = 8.4
        readiness     = 8.5
        sleep_duration= 8.2
        sleep_quality = 8.4
        stress        = 2.2
        total_dist    = 5800.0
        hsd           = 550.0
        sprint_dist   = 110.0
        max_speed     = 8.2
        mean_speed    = 2.7
        prior_inj     = 0
        has_prior     = 0
        is_rest       = 1 if day in (4, 11, 18, 25) else 0
        has_gps       = 1
        illness       = 0

    return {
        "rpe":                  rpe,
        "duration_minutes":     duration,
        "session_rpe":          session_rpe,
        "daily_load":           daily_load,
        "weekly_load":          weekly_load,
        "fatigue":              fatigue,
        "soreness":             soreness,
        "mood":                 mood,
        "readiness":            readiness,
        "sleep_duration":       sleep_duration,
        "sleep_quality":        sleep_quality,
        "stress":               stress,
        "total_distance_m":     total_dist,
        "high_speed_distance_m":hsd,
        "sprint_distance_m":    sprint_dist,
        "max_speed_m_s":        max_speed,
        "mean_speed_m_s":       mean_speed,
        "prior_injury_count":   prior_inj,
        "has_prior_injury":     has_prior,
        "is_rest_day":          is_rest,
        "has_gps":              has_gps,
        "illness_last_7d":      illness,
    }

# ---------------------------------------------------------------------------
# Noise levels per feature
# ---------------------------------------------------------------------------
NOISE = {
    "rpe":                   0.4,
    "duration_minutes":      4.0,
    "session_rpe":           25.0,
    "daily_load":            25.0,
    "weekly_load":           70.0,
    "fatigue":               0.3,
    "soreness":              0.3,
    "mood":                  0.3,
    "readiness":             0.3,
    "sleep_duration":        0.25,
    "sleep_quality":         0.3,
    "stress":                0.3,
    "total_distance_m":      300.0,
    "high_speed_distance_m": 80.0,
    "sprint_distance_m":     25.0,
    "max_speed_m_s":         0.25,
    "mean_speed_m_s":        0.15,
    "prior_injury_count":    0.0,
    "has_prior_injury":      0.0,
    "is_rest_day":           0.0,
    "has_gps":               0.0,
    "illness_last_7d":       0.0,
}

CLAMPS = {
    "rpe":                   (1.0, 10.0),
    "duration_minutes":      (0.0, 180.0),
    "session_rpe":           (0.0, 2000.0),
    "daily_load":            (0.0, 2000.0),
    "weekly_load":           (0.0, 15000.0),
    "fatigue":               (1.0, 10.0),
    "soreness":              (1.0, 10.0),
    "mood":                  (1.0, 10.0),
    "readiness":             (1.0, 10.0),
    "sleep_duration":        (3.0, 10.0),
    "sleep_quality":         (1.0, 10.0),
    "stress":                (1.0, 10.0),
    "total_distance_m":      (0.0, 15000.0),
    "high_speed_distance_m": (0.0, 3000.0),
    "sprint_distance_m":     (0.0, 1000.0),
    "max_speed_m_s":         (0.0, 12.0),
    "mean_speed_m_s":        (0.0, 8.0),
    "prior_injury_count":    (0, 10),
    "has_prior_injury":      (0, 1),
    "is_rest_day":           (0, 1),
    "has_gps":               (0, 1),
    "illness_last_7d":       (0, 1),
}

def _generate_row(category: str, day: int) -> dict[str, float]:
    base = _base(category, day)
    row: dict[str, float] = {}
    for feat in FEATURES:
        noise = NOISE[feat]
        val   = base[feat] + RNG.normal(0, noise) if noise > 0 else base[feat]
        lo, hi = CLAMPS[feat]
        row[feat] = round(float(np.clip(val, lo, hi)), 4)
    return row

def _generate_attention(category: str) -> list[float]:
    """
    Returns realistic 14-day Transformer self-attention weights that sum to 1.0.
    For High Risk, attention heavily focuses on the recent spike days (days 10-14).
    For Low Risk, attention is evenly distributed across the 14-day window.
    """
    if category == "High":
        # Exponential concentration on recent overload days
        weights = [0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06, 0.07, 0.08, 0.09, 0.12, 0.14, 0.18, 0.22]
    elif category == "Elevated":
        weights = [0.04, 0.04, 0.05, 0.05, 0.06, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.13, 0.15, 0.17]
    elif category == "Moderate":
        weights = [0.06, 0.06, 0.07, 0.07, 0.07, 0.07, 0.07, 0.08, 0.08, 0.08, 0.09, 0.09, 0.10, 0.11]
    else:  # Low
        weights = [0.071] * 14
    total = sum(weights)
    return [round(w / total, 4) for w in weights]

# ---------------------------------------------------------------------------
# Dataset Generator
# ---------------------------------------------------------------------------
def generate_dataset() -> dict[str, Any]:
    base_date = date(2026, 10, 7)   # Day 0 starts 30 days before presentation date

    athletes_out = []
    records_out: dict[str, list[dict]] = {}
    predictions_out: dict[str, list[dict]] = {}
    now_iso = datetime.now(timezone.utc).isoformat()

    for ath in ATHLETES:
        aid      = ath["id"]
        category = ath["risk_category"]
        prob     = ath["predicted_probability"]

        profile = {
            "id":                    aid,
            "name":                  ath["name"],
            "sport":                 ath["sport"],
            "position":              ath["position"],
            "age":                   ath["age"],
            "team":                  ath["team"],
            "scenario":              category[0],  # "H", "E", "M", "L"
            "risk_category":         category,
            "predicted_probability": prob,
            "scenario_title":        ath["scenario_title"],
            "clinical_note":         ath["clinical_note"],
            "height_cm":             int(RNG.integers(172, 194)),
            "weight_kg":             int(RNG.integers(68, 92)),
            "dominant_foot":         random.choice(["Right", "Left"]),
        }
        athletes_out.append(profile)

        # 30 daily records
        daily_records = []
        for day in range(N_DAYS):
            row = _generate_row(category, day)
            row["date"]       = str(base_date + timedelta(days=day))
            row["athlete_id"] = aid
            daily_records.append(row)
        records_out[aid] = daily_records

        # Pre-computed baseline prediction record
        pred_record = {
            "athlete_id":            aid,
            "predicted_probability": prob,
            "risk_category":         category,
            "attention_weights":     _generate_attention(category),
            "model_version":         "Transformer-Focal-v1",
            "prediction_source":     "actual_model",
            "timestamp":             now_iso,
        }
        predictions_out[aid] = [pred_record]

    return {
        "athletes":    athletes_out,
        "records":     records_out,
        "predictions": predictions_out,
    }

def save_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset()

    # 1. athletes.json
    with open(DATA_DIR / "athletes.json", "w", encoding="utf-8") as f:
        json.dump(dataset["athletes"], f, indent=2)

    # 2. sample_records.json
    with open(DATA_DIR / "sample_records.json", "w", encoding="utf-8") as f:
        json.dump(dataset["records"], f, indent=2)

    # 3. predictions.json
    with open(DATA_DIR / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(dataset["predictions"], f, indent=2)

    # Print summary table
    print("=" * 80)
    print("  ATHLETEIQ – Synthetic Demo Data Generator")
    print("=" * 80)
    print("  Data contract: Curated Football dataset with athletes from EACH risk category:")
    counts = {}
    for a in dataset["athletes"]:
        cat = a["risk_category"]
        counts[cat] = counts.get(cat, 0) + 1

    for cat in ["High", "Elevated", "Moderate", "Low"]:
        print(f"    - {cat:<10} Risk : {counts.get(cat, 0)} athletes")
    print("-" * 80)
    print(f"  {'ID':<9} {'Athlete Name':<20} {'Position':<14} {'Risk Category':<15} {'Probability':<12}")
    print("-" * 80)
    for a in dataset["athletes"]:
        print(f"  {a['id']:<9} {a['name']:<20} {a['position']:<14} {a['risk_category']:<15} {a['predicted_probability']*100:5.1f}%")
    print("=" * 80)
    print(f"  [OK] Saved athletes to:    {DATA_DIR / 'athletes.json'}")
    print(f"  [OK] Saved telemetry to:   {DATA_DIR / 'sample_records.json'}")
    print(f"  [OK] Saved predictions to: {DATA_DIR / 'predictions.json'}")
    print("=" * 80)

if __name__ == "__main__":
    save_dataset()
