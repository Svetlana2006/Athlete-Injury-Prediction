"""
Athlete Injury-Risk Data Pipeline
==================================
Reads raw SoccerMon data (subjective CSVs, session JSON, GPS summaries,
injury/illness records), melts wide-format files into long format, merges
all sources, engineers analytical features, and outputs a single
daily_observations.csv ready for ML training and database seeding.

Usage:
    python data/pipeline.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SUBJECTIVE_DIR = BASE_DIR / "subjective"
DERIVED_DIR = BASE_DIR / "derived"
OUTPUT_DIR = BASE_DIR / "processed"

HORIZON_DAYS = 7  # injury prediction window


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING — read every raw source and normalise to long format
# ═══════════════════════════════════════════════════════════════════════════

def _melt_wide_csv(filepath: Path, value_name: str) -> pd.DataFrame:
    """Convert a SoccerMon wide-format CSV (Date × Players) → long format.

    Input shape:   Date | Player-UUID-1 | Player-UUID-2 | …
    Output shape:  date | athlete_id    | <value_name>
    """
    df = pd.read_csv(filepath)
    date_col = df.columns[0]                       # "Date", "Fatigue Data", …
    melted = df.melt(
        id_vars=[date_col], var_name="athlete_id", value_name=value_name
    )
    melted = melted.rename(columns={date_col: "date"})
    melted["date"] = pd.to_datetime(melted["date"], format="%d.%m.%Y")
    melted = melted.dropna(subset=[value_name])     # drop days with no report
    return melted


def load_training_load() -> pd.DataFrame:
    """Load training-load data from session.json + 8 pre-computed wide CSVs."""
    tl_dir = SUBJECTIVE_DIR / "training-load"

    # ── Session-level data (RPE, duration, sRPE) from JSON ──
    with open(tl_dir / "session.json", encoding="utf-8") as f:
        sessions_raw: dict = json.load(f)

    rows: list[dict] = []
    for athlete_id, sessions in sessions_raw.items():
        for s in sessions:
            rows.append({
                "athlete_id": athlete_id,
                "date": pd.to_datetime(s["date"], format="%d.%m.%Y"),
                "rpe": float(s["rpe"]),
                "duration_minutes": float(s["duration"]),
                "session_rpe": float(s["srpe"]),
            })
    sessions = pd.DataFrame(rows)

    # Aggregate multiple sessions on the same day
    sessions_daily = (
        sessions
        .groupby(["athlete_id", "date"])
        .agg(
            rpe=("rpe", "max"),
            duration_minutes=("duration_minutes", "sum"),
            session_rpe=("session_rpe", "sum"),
            session_count=("rpe", "count"),
        )
        .reset_index()
    )

    # ── Pre-computed workload CSVs ──
    metric_files = {
        "daily_load":  "daily_load.csv",
        "acwr":        "acwr.csv",
        "atl":         "atl.csv",
        "ctl28":       "ctl28.csv",
        "ctl42":       "ctl42.csv",
        "monotony":    "monotony.csv",
        "strain":      "strain.csv",
        "weekly_load": "weekly_load.csv",
    }

    result = sessions_daily
    for value_name, filename in metric_files.items():
        fp = tl_dir / filename
        if fp.exists():
            melted = _melt_wide_csv(fp, value_name)
            result = result.merge(melted, on=["athlete_id", "date"], how="outer")

    return result


def load_wellness() -> pd.DataFrame:
    """Load 7 daily wellness self-report CSVs and merge into one frame."""
    w_dir = SUBJECTIVE_DIR / "wellness"

    wellness_files = {
        "fatigue":        "fatigue.csv",
        "soreness":       "soreness.csv",
        "mood":           "mood.csv",
        "readiness":      "readiness.csv",
        "sleep_duration": "sleep_duration.csv",
        "sleep_quality":  "sleep_quality.csv",
        "stress":         "stress.csv",
    }

    result: pd.DataFrame | None = None
    for value_name, filename in wellness_files.items():
        fp = w_dir / filename
        if not fp.exists():
            continue
        melted = _melt_wide_csv(fp, value_name)
        result = melted if result is None else result.merge(
            melted, on=["athlete_id", "date"], how="outer"
        )

    return result if result is not None else pd.DataFrame()


def load_gps() -> pd.DataFrame:
    """Load GPS summary CSVs and aggregate to one row per athlete per day."""
    gps_files = sorted(DERIVED_DIR.glob("*_gps_summary.csv"))
    if not gps_files:
        print("      ⚠  No GPS summary files found in data/derived/")
        return pd.DataFrame()

    frames = [pd.read_csv(f) for f in gps_files]
    gps = pd.concat(frames, ignore_index=True)
    gps = gps.rename(columns={"session_date": "date"})
    gps["date"] = pd.to_datetime(gps["date"])

    # Aggregate multiple GPS sessions on the same day
    gps_daily = (
        gps
        .groupby(["athlete_id", "date"])
        .agg(
            gps_samples=("gps_samples", "sum"),
            max_speed_m_s=("max_speed_m_s", "max"),
            total_distance_m=("total_distance_m", "sum"),
            high_speed_distance_m=("high_speed_distance_m", "sum"),
            sprint_distance_m=("sprint_distance_m", "sum"),
            mean_speed_m_s=("mean_speed_m_s", "mean"),
        )
        .reset_index()
    )

    return gps_daily


def load_injuries() -> pd.DataFrame:
    """Load and parse injury records (JSON body-area → severity mapping)."""
    fp = SUBJECTIVE_DIR / "injury" / "injury.csv"
    df = pd.read_csv(fp)

    records: list[dict] = []
    for _, row in df.iterrows():
        try:
            type_info: dict = json.loads(row["type"])
            for body_area, severity in type_info.items():
                records.append({
                    "athlete_id": row["player_name"],
                    "date": pd.to_datetime(row["timestamp"], format="%d.%m.%Y"),
                    "body_area": body_area,
                    "severity": severity,
                })
        except (json.JSONDecodeError, AttributeError, TypeError):
            records.append({
                "athlete_id": row["player_name"],
                "date": pd.to_datetime(row["timestamp"], format="%d.%m.%Y"),
                "body_area": "unknown",
                "severity": "unknown",
            })

    return pd.DataFrame(records)


def load_illness() -> pd.DataFrame:
    """Load illness records."""
    fp = SUBJECTIVE_DIR / "illness" / "illness.csv"
    df = pd.read_csv(fp)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%d.%m.%Y")
    df = df.rename(columns={"player_name": "athlete_id", "timestamp": "date"})
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING — derive every metric from the approved plan
# ═══════════════════════════════════════════════════════════════════════════

def engineer_features(
    merged: pd.DataFrame,
    injuries: pd.DataFrame,
    illness: pd.DataFrame,
) -> pd.DataFrame:
    """Compute all derived analytical features and perform sports-science domain imputation.

    Groups of features:
      • Workload  — load change %, validation of ACWR, rest-day imputation
      • Wellness  — forward-fill timeline continuity, baseline imputation, composite score
      • GPS       — non-monitored day zero-filling, HSR ratio, distance change %
      • History   — prior injury count, days since last injury (730d uninjured baseline),
                    illness flag, days since rest
      • Target    — injury_next_7d (binary label)
    """
    df = merged.sort_values(["athlete_id", "date"]).reset_index(drop=True)

    # ── Team label ──
    df["team"] = df["athlete_id"].str.extract(r"^(Team[AB])", expand=False)

    # ────────────────────────────── WORKLOAD ──────────────────────────────
    # Rest days have 0 duration and 0 session RPE
    session_cols = ["duration_minutes", "session_rpe", "session_count", "rpe"]
    for c in session_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # Pre-computed workload metrics (daily_load, acwr, atl, ctl, monotony, strain)
    load_metrics = [
        "daily_load", "acwr", "atl", "ctl28", "ctl42", "monotony", "strain", "weekly_load"
    ]
    for c in load_metrics:
        if c in df.columns:
            df[c] = df.groupby("athlete_id")[c].ffill()
            df[c] = df[c].fillna(0.0)

    # Explicit rest day flag (load == 0)
    df["is_rest_day"] = ((df["daily_load"] == 0.0) & (df["duration_minutes"] == 0.0)).astype(int)

    # Week-over-week load change % (fill baseline zeros with 0.0)
    rolling_7d = df.groupby("athlete_id")["daily_load"].transform(
        lambda x: x.rolling(7, min_periods=1).sum()
    )
    prev_7d = df.groupby("athlete_id")["daily_load"].transform(
        lambda x: x.rolling(7, min_periods=1).sum().shift(7)
    )
    df["load_change_pct"] = (
        (rolling_7d - prev_7d) / prev_7d.replace(0, np.nan) * 100
    ).fillna(0.0).clip(-100.0, 500.0)

    # ──────────────────────────── GPS / MOVEMENT ─────────────────────────
    # On non-GPS / rest days, movement metrics are 0.0
    gps_cols = [
        "gps_samples", "max_speed_m_s", "total_distance_m",
        "high_speed_distance_m", "sprint_distance_m", "mean_speed_m_s"
    ]
    for c in gps_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    df["has_gps"] = (df["total_distance_m"] > 0).astype(int)

    # High-speed running ratio
    denom_dist = df["total_distance_m"].replace(0, np.nan)
    df["hsr_ratio"] = (df["high_speed_distance_m"] / denom_dist).fillna(0.0)

    # 7-day distance change %
    dist_7d = df.groupby("athlete_id")["total_distance_m"].transform(
        lambda x: x.rolling(7, min_periods=1).sum()
    )
    prev_dist = df.groupby("athlete_id")["total_distance_m"].transform(
        lambda x: x.rolling(7, min_periods=1).sum().shift(7)
    )
    df["distance_change_pct"] = (
        (dist_7d - prev_dist) / prev_dist.replace(0, np.nan) * 100
    ).fillna(0.0).clip(-100.0, 500.0)

    # ────────────────────────────── WELLNESS ──────────────────────────────
    # Wellness carryover: forward-fill recent physical state up to 7 days,
    # then athlete baseline median, then default neutral baseline (5.0, sleep 8.0)
    wellness_base_cols = [
        "fatigue", "soreness", "mood", "readiness", "sleep_duration", "sleep_quality", "stress"
    ]
    for c in wellness_base_cols:
        if c in df.columns:
            df[c] = df.groupby("athlete_id")[c].ffill()
            ath_median = df.groupby("athlete_id")[c].transform("median")
            df[c] = df[c].fillna(ath_median)
            default_val = 8.0 if c == "sleep_duration" else 5.0
            df[c] = df[c].fillna(default_val)

    # Composite wellness score (1-10)
    core_wellness = ["fatigue", "soreness", "mood", "readiness", "sleep_quality", "stress"]
    df["composite_wellness"] = df[core_wellness].mean(axis=1)

    # Sleep deficit (hours below 8h)
    df["sleep_deficit"] = (8.0 - df["sleep_duration"]).clip(lower=0)

    # Rolling 7-day trends for fatigue and soreness
    for col in ("fatigue", "soreness"):
        df[f"{col}_trend_7d"] = df.groupby("athlete_id")[col].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )

    # Wellness delta (today vs athlete's 28-day personal rolling baseline)
    baseline_28d = df.groupby("athlete_id")["composite_wellness"].transform(
        lambda x: x.rolling(28, min_periods=1).mean()
    )
    df["wellness_delta"] = (df["composite_wellness"] - baseline_28d).fillna(0.0)

    # ────────────────────────────── HISTORY ───────────────────────────────
    inj_events = injuries[["athlete_id", "date"]].drop_duplicates()

    df["prior_injury_count"] = 0
    # Clean uninjured baseline: 730 days (2 years injury-free)
    df["days_since_last_injury"] = 730.0

    for athlete_id in df["athlete_id"].unique():
        a_mask = df["athlete_id"] == athlete_id
        a_inj_dates = sorted(
            inj_events.loc[inj_events["athlete_id"] == athlete_id, "date"].tolist()
        )
        if not a_inj_dates:
            continue
        for idx in df.loc[a_mask].index:
            cur = df.loc[idx, "date"]
            past = [d for d in a_inj_dates if d < cur]
            df.loc[idx, "prior_injury_count"] = len(past)
            if past:
                df.loc[idx, "days_since_last_injury"] = float((cur - max(past)).days)

    df["has_prior_injury"] = (df["prior_injury_count"] > 0).astype(int)

    # Illness flag: athlete was sick within the previous 7 days
    illness_events = illness[["athlete_id", "date"]].drop_duplicates()
    df["illness_last_7d"] = 0
    for _, row in illness_events.iterrows():
        mask = (
            (df["athlete_id"] == row["athlete_id"])
            & (df["date"] >= row["date"])
            & (df["date"] <= row["date"] + pd.Timedelta(days=7))
        )
        df.loc[mask, "illness_last_7d"] = 1

    # Days since rest — consecutive training days (load > 0)
    df["days_since_rest"] = 0
    for _, grp in df.groupby("athlete_id"):
        streak = 0
        for idx in grp.index:
            val = df.loc[idx, "daily_load"]
            streak = streak + 1 if (pd.notna(val) and val > 0) else 0
            df.loc[idx, "days_since_rest"] = streak

    # ────────────────────────────── TARGET ────────────────────────────────
    # Step 1: Default label = 0 (healthy)
    df["injury_next_7d"] = 0

    # Step 2: Label the 7 days *before* each injury as positive (1)
    for _, inj_row in inj_events.iterrows():
        pre_injury_mask = (
            (df["athlete_id"] == inj_row["athlete_id"])
            & (df["date"] >= inj_row["date"] - pd.Timedelta(days=HORIZON_DAYS))
            & (df["date"] < inj_row["date"])
        )
        df.loc[pre_injury_mask, "injury_next_7d"] = 1

    # Step 3: Blackout — NaN out the injury date itself and the 2 days after.
    # These rows have forward-filled wellness/GPS from *after* the injury
    # event, making them contaminated.  Setting NaN tells the dataset builder
    # to drop them, preventing label leakage into training.
    for _, inj_row in inj_events.iterrows():
        blackout_mask = (
            (df["athlete_id"] == inj_row["athlete_id"])
            & (df["date"] >= inj_row["date"])
            & (df["date"] <= inj_row["date"] + pd.Timedelta(days=2))
        )
        df.loc[blackout_mask, "injury_next_7d"] = np.nan

    n_blacked_out = int(df["injury_next_7d"].isna().sum())
    print(f"      Blackout rows (injury day + 2d post-injury): {n_blacked_out}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 3. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 62)
    print("   ATHLETE INJURY-RISK DATA PIPELINE")
    print("=" * 62)

    # ── Load ──
    print("\n[1/6] Loading training-load data …")
    training = load_training_load()
    n_tl = training["athlete_id"].nunique()
    print(f"      {len(training):>7,} records  ·  {n_tl} athletes")

    print("\n[2/6] Loading wellness data …")
    wellness = load_wellness()
    n_wl = wellness["athlete_id"].nunique() if not wellness.empty else 0
    print(f"      {len(wellness):>7,} records  ·  {n_wl} athletes")

    print("\n[3/6] Loading GPS summaries …")
    gps = load_gps()
    n_gps = gps["athlete_id"].nunique() if not gps.empty else 0
    print(f"      {len(gps):>7,} records  ·  {n_gps} athletes")

    print("\n[4/6] Loading injury & illness data …")
    injuries = load_injuries()
    illness = load_illness()
    print(f"      {len(injuries)} injury events  ·  {len(illness)} illness events")

    # ── Merge ──
    print("\n[5/6] Merging all sources …")
    merged = training.merge(wellness, on=["athlete_id", "date"], how="outer")
    if not gps.empty:
        merged = merged.merge(gps, on=["athlete_id", "date"], how="outer")
    n_all = merged["athlete_id"].nunique()
    print(f"      {len(merged):>7,} merged rows  ·  {n_all} athletes")

    # ── Engineer ──
    print("\n[6/6] Engineering features …")
    final = engineer_features(merged, injuries, illness)

    # ── Save ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    obs_path = OUTPUT_DIR / "daily_observations.csv"
    inj_path = OUTPUT_DIR / "injuries.csv"
    final.to_csv(obs_path, index=False)
    injuries.to_csv(inj_path, index=False)

    # ── Summary ──
    pos = int(final["injury_next_7d"].sum())
    neg = len(final) - pos
    pct = pos / len(final) * 100 if len(final) else 0

    print(f"\n{'=' * 62}")
    print(f"   PIPELINE COMPLETE")
    print(f"{'=' * 62}")
    print(f"   Output .............. {obs_path}")
    print(f"   Rows ................ {len(final):,}")
    print(f"   Athletes ............ {final['athlete_id'].nunique()}")
    print(f"   Date range .......... {final['date'].min().date()}"
          f" -> {final['date'].max().date()}")
    print(f"   Injury target ....... {pos:,} positive ({pct:.1f}%)"
          f" / {neg:,} negative")
    print(f"\n   Column coverage ({len(final.columns)} columns):")
    for col in final.columns:
        nn = final[col].notna().sum()
        p = nn / len(final) * 100
        print(f"     {col:<28s}  {nn:>7,} non-null  ({p:5.1f}%)")


if __name__ == "__main__":
    main()
