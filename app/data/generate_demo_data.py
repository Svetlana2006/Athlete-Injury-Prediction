"""
ATHLETEIQ – Synthetic Demo Data Generator
==========================================
Generates 14 fictional football athletes with 30 days of realistic sensor records.
Records use the EXACT same 22 feature columns the trained Transformer expects.

Five scenario archetypes are distributed across the athlete roster:
  A – Stable / Low risk
  B – Increasing workload (rising ACWR trend)
  C – Degrading recovery (fatigue accumulation)
  D – Fluctuating measurements (high variance)
  E – Improving trend (recent load reduction)

Each athlete gets EXACTLY 30 days of records. The model uses a 14-day
lookback window, so the most recent 14 rows are passed to the model.
"""

from __future__ import annotations

import json
import random
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
# Feature columns (must match app/ml/train.py RAW_FEATURE_COLUMNS exactly)
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
# Athlete roster definitions
# ---------------------------------------------------------------------------
ATHLETES = [
    # --- Football (Apex FC) ---
    {"id": "ATH-101", "name": "Marcus Sterling",   "sport": "Football", "position": "Winger",       "age": 24, "team": "Apex FC",    "scenario": "B"},
    {"id": "ATH-102", "name": "Rashid Patel",       "sport": "Football", "position": "Center-Back",  "age": 26, "team": "Apex FC",    "scenario": "A"},
    {"id": "ATH-103", "name": "Liam Vance",         "sport": "Football", "position": "Center-Back",  "age": 28, "team": "Apex FC",    "scenario": "A"},
    {"id": "ATH-104", "name": "Elena Rostova",      "sport": "Football", "position": "Midfielder",   "age": 22, "team": "Apex FC",    "scenario": "A"},
    {"id": "ATH-105", "name": "Noah Becker",        "sport": "Football", "position": "Striker",      "age": 25, "team": "Apex FC",    "scenario": "D"},
    {"id": "ATH-106", "name": "Devraj Sharma",      "sport": "Football", "position": "Goalkeeper",   "age": 23, "team": "Apex FC",    "scenario": "A"},
    {"id": "ATH-107", "name": "Mateo Silva",        "sport": "Football", "position": "Midfielder",   "age": 27, "team": "Apex FC",    "scenario": "B"},
    {"id": "ATH-108", "name": "Arjun Nair",         "sport": "Football", "position": "Full-Back",    "age": 27, "team": "Apex FC",    "scenario": "A"},
    {"id": "ATH-109", "name": "Kasper Lind",        "sport": "Football", "position": "Center-Back",  "age": 26, "team": "Apex FC",    "scenario": "E"},
    {"id": "ATH-110", "name": "Tariq Mansoor",      "sport": "Football", "position": "Winger",       "age": 25, "team": "Apex FC",    "scenario": "C"},
    {"id": "ATH-111", "name": "Lucas Romero",       "sport": "Football", "position": "Striker",      "age": 30, "team": "Apex FC",    "scenario": "A"},
    {"id": "ATH-112", "name": "Bukayo Saka",        "sport": "Football", "position": "Winger",       "age": 22, "team": "Apex FC",    "scenario": "E"},
    {"id": "ATH-113", "name": "Fernanda Diaz",      "sport": "Football", "position": "Midfielder",   "age": 24, "team": "Apex FC",    "scenario": "A"},
    {"id": "ATH-114", "name": "Owen Griffiths",     "sport": "Football", "position": "Full-Back",    "age": 29, "team": "Apex FC",    "scenario": "D"},
]

assert len(ATHLETES) == 14, "The demo roster must contain only football athletes"

# ---------------------------------------------------------------------------
# Scenario base values (before noise)
# ---------------------------------------------------------------------------
# Each dict contains (mean, std) per feature for each scenario archetype.
# Values are in original / natural units (unscaled).

def _base(scenario: str, day: int) -> dict[str, float]:
    """
    Returns target values for each feature for a given scenario and day index
    (0 = oldest, N_DAYS-1 = most recent).
    """
    t = day / (N_DAYS - 1)   # 0.0 → 1.0 progress through the monitoring period

    if scenario == "A":   # Stable / low risk
        rpe           = 5.0
        duration      = 70.0
        session_rpe   = rpe * duration
        daily_load    = session_rpe
        weekly_load   = daily_load * 5
        fatigue       = 3.0
        soreness      = 2.0
        mood          = 7.5
        readiness     = 7.5
        sleep_duration= 7.8
        sleep_quality = 7.5
        stress        = 3.0
        total_dist    = 7000.0
        hsd           = 900.0
        sprint_dist   = 200.0
        max_speed     = 8.5
        mean_speed    = 3.0
        prior_inj     = 0
        has_prior     = 0
        is_rest       = 0
        has_gps       = 1
        illness       = 0

    elif scenario == "B":   # Increasing workload
        rpe           = 5.0 + 3.5 * t
        duration      = 65.0 + 30.0 * t
        session_rpe   = rpe * duration
        daily_load    = session_rpe
        weekly_load   = daily_load * (4.5 + 1.5 * t)
        fatigue       = 3.5 + 3.5 * t
        soreness      = 3.0 + 3.5 * t
        mood          = 7.5 - 2.5 * t
        readiness     = 7.5 - 3.0 * t
        sleep_duration= 7.5 - 1.0 * t
        sleep_quality = 7.5 - 2.0 * t
        stress        = 3.0 + 4.0 * t
        total_dist    = 7000.0 + 4000.0 * t
        hsd           = 900.0 + 600.0 * t
        sprint_dist   = 200.0 + 200.0 * t
        max_speed     = 8.5 + 1.0 * t
        mean_speed    = 3.0 + 0.5 * t
        prior_inj     = 1
        has_prior     = 1
        is_rest       = 0
        has_gps       = 1
        illness       = 0

    elif scenario == "C":   # Degrading recovery / fatigue accumulation
        rpe           = 5.5
        duration      = 75.0
        session_rpe   = rpe * duration
        daily_load    = session_rpe
        weekly_load   = daily_load * 5
        fatigue       = 4.0 + 4.5 * t
        soreness      = 4.0 + 4.5 * t
        mood          = 7.0 - 3.5 * t
        readiness     = 7.0 - 4.0 * t
        sleep_duration= 8.0 - 2.5 * t
        sleep_quality = 7.0 - 3.5 * t
        stress        = 4.0 + 4.0 * t
        total_dist    = 6500.0
        hsd           = 800.0
        sprint_dist   = 180.0
        max_speed     = 8.0
        mean_speed    = 2.8
        prior_inj     = 2
        has_prior     = 1
        is_rest       = 0
        has_gps       = 1
        illness       = 1 if t > 0.7 else 0

    elif scenario == "D":   # Fluctuating / high variance
        phase         = np.sin(day * 0.6) * 0.5 + 0.5
        rpe           = 4.0 + 5.0 * phase
        duration      = 55.0 + 50.0 * phase
        session_rpe   = rpe * duration
        daily_load    = session_rpe
        weekly_load   = daily_load * (4.0 + 2.0 * phase)
        fatigue       = 3.0 + 5.0 * phase
        soreness      = 3.0 + 5.0 * phase
        mood          = 8.0 - 4.0 * phase
        readiness     = 8.0 - 4.0 * phase
        sleep_duration= 7.0 + 2.0 * phase
        sleep_quality = 7.0 - 3.0 * phase
        stress        = 3.0 + 4.0 * phase
        total_dist    = 5000.0 + 5000.0 * phase
        hsd           = 700.0 + 700.0 * phase
        sprint_dist   = 150.0 + 250.0 * phase
        max_speed     = 7.5 + 2.5 * phase
        mean_speed    = 2.5 + 1.5 * phase
        prior_inj     = 1
        has_prior     = 1
        is_rest       = 1 if phase < 0.2 else 0
        has_gps       = 1 if phase > 0.1 else 0
        illness       = 0

    else:   # scenario E – Improving / recovering
        rpe           = 7.5 - 3.0 * t
        duration      = 90.0 - 25.0 * t
        session_rpe   = rpe * duration
        daily_load    = session_rpe
        weekly_load   = daily_load * (5.5 - 1.5 * t)
        fatigue       = 7.0 - 4.0 * t
        soreness      = 7.0 - 4.5 * t
        mood          = 5.0 + 3.0 * t
        readiness     = 5.0 + 3.5 * t
        sleep_duration= 6.0 + 2.0 * t
        sleep_quality = 5.5 + 2.5 * t
        stress        = 7.0 - 4.0 * t
        total_dist    = 9500.0 - 3000.0 * t
        hsd           = 1300.0 - 500.0 * t
        sprint_dist   = 350.0 - 170.0 * t
        max_speed     = 9.5 - 1.0 * t
        mean_speed    = 3.5 - 0.5 * t
        prior_inj     = 2
        has_prior     = 1
        is_rest       = 1 if t < 0.1 or (0.4 < t < 0.5) else 0
        has_gps       = 1
        illness       = 1 if t < 0.15 else 0

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
    "rpe":                   0.5,
    "duration_minutes":      5.0,
    "session_rpe":           30.0,
    "daily_load":            30.0,
    "weekly_load":           80.0,
    "fatigue":               0.4,
    "soreness":              0.4,
    "mood":                  0.4,
    "readiness":             0.4,
    "sleep_duration":        0.3,
    "sleep_quality":         0.4,
    "stress":                0.4,
    "total_distance_m":      400.0,
    "high_speed_distance_m": 100.0,
    "sprint_distance_m":     30.0,
    "max_speed_m_s":         0.3,
    "mean_speed_m_s":        0.2,
    "prior_injury_count":    0.0,   # integer – no noise
    "has_prior_injury":      0.0,   # binary
    "is_rest_day":           0.0,   # binary
    "has_gps":               0.0,   # binary
    "illness_last_7d":       0.0,   # binary
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


def _generate_row(scenario: str, day: int) -> dict[str, float]:
    base = _base(scenario, day)
    row: dict[str, float] = {}
    for feat in FEATURES:
        noise = NOISE[feat]
        val   = base[feat] + RNG.normal(0, noise) if noise > 0 else base[feat]
        lo, hi = CLAMPS[feat]
        row[feat] = round(float(np.clip(val, lo, hi)), 4)
    return row


# ---------------------------------------------------------------------------
# Full dataset generation
# ---------------------------------------------------------------------------

def generate_dataset() -> dict[str, Any]:
    """
    Returns a dict:
        {
            "athletes": [ athlete_profile, ... ],
            "records":  { "ATH-101": [ { date, feature... }, ... ], ... }
        }
    """
    from datetime import date, timedelta

    base_date = date(2026, 10, 7)   # day 0 is 30 days before demo day

    athletes_out = []
    records_out: dict[str, list[dict]] = {}

    for ath in ATHLETES:
        ath_id   = ath["id"]
        scenario = ath["scenario"]

        profile = {
            "id":          ath_id,
            "name":        ath["name"],
            "sport":       ath["sport"],
            "position":    ath["position"],
            "age":         ath["age"],
            "team":        ath["team"],
            "scenario":    scenario,
            "height_cm":   int(RNG.integers(170, 196)),
            "weight_kg":   int(RNG.integers(65, 95)),
            "dominant_foot": random.choice(["Right", "Left"]),
        }
        athletes_out.append(profile)

        daily_records = []
        for day in range(N_DAYS):
            row = _generate_row(scenario, day)
            row["date"]       = str(base_date + timedelta(days=day))
            row["athlete_id"] = ath_id
            daily_records.append(row)

        records_out[ath_id] = daily_records

    return {"athletes": athletes_out, "records": records_out}


def save_dataset() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset()

    with open(DATA_DIR / "athletes.json", "w") as f:
        json.dump(dataset["athletes"], f, indent=2)

    with open(DATA_DIR / "sample_records.json", "w") as f:
        json.dump(dataset["records"], f, indent=2)

    print(f"[DataGen] Saved {len(dataset['athletes'])} athletes to {DATA_DIR}")


if __name__ == "__main__":
    save_dataset()
