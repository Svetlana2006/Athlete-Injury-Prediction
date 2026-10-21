"""
audit_dataset.py
================
Dataset audit script adapted from partner's code to match the actual
SoccerMon dataset structure in Athlete-Injury-Prediction.

Audits:
1. Raw / Unaggregated source data (sessions, raw GPS logs, raw injuries)
2. Normalized table-level source data (training load, wellness, GPS daily, injuries)
3. Duplicate athlete-day keys across all tables
4. Overlap & intersections between tables (inner join vs outer join)
5. Training rows matching / missing wellness and GPS
6. Injury event counts, injured athletes, duplicate athlete-injury dates
"""

import json
from pathlib import Path
import pandas as pd

from data.pipeline import (
    BASE_DIR,
    SUBJECTIVE_DIR,
    DERIVED_DIR,
    OUTPUT_DIR,
    load_training_load,
    load_wellness,
    load_gps,
    load_injuries,
)

KEYS = ["athlete_id", "date"]

def audit():
    print("=" * 65)
    print("   DATASET AUDIT REPORT (SoccerMon / Athlete-Injury-Prediction)")
    print("=" * 65)

    # ─────────────────────────────────────────────────────────────
    # PART 1: RAW SOURCE FILES (Before daily aggregation / melting)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PART 1: RAW SESSION-LEVEL / UNAGGREGATED LOGS")
    print("=" * 65)

    # 1.1 Raw Training Sessions from session.json
    session_json_path = SUBJECTIVE_DIR / "training-load" / "session.json"
    if session_json_path.exists():
        with open(session_json_path, encoding="utf-8") as f:
            sessions_raw = json.load(f)
        raw_session_rows = []
        for athlete_id, sessions in sessions_raw.items():
            for s in sessions:
                raw_session_rows.append({
                    "athlete_id": athlete_id,
                    "date": pd.to_datetime(s["date"], format="%d.%m.%Y"),
                    "rpe": float(s["rpe"]),
                    "duration_minutes": float(s["duration"]),
                    "session_rpe": float(s["srpe"]),
                })
        df_raw_sessions = pd.DataFrame(raw_session_rows)
        df_raw_sessions["date"] = df_raw_sessions["date"].dt.normalize()
        dup_raw_sessions = df_raw_sessions.duplicated(KEYS).sum()
        print(f"\n--- RAW TRAINING SESSIONS (session.json) ---")
        print(f"Total session logs: {len(df_raw_sessions):,}")
        print(f"Unique athletes: {df_raw_sessions['athlete_id'].nunique()}")
        print(f"Duplicate athlete-date entries (multiple sessions on same day): {dup_raw_sessions:,}")

    # 1.2 Raw GPS Summary files (before daily grouping)
    gps_files = sorted(DERIVED_DIR.glob("*_gps_summary.csv"))
    if gps_files:
        raw_gps_frames = [pd.read_csv(f) for f in gps_files]
        df_raw_gps = pd.concat(raw_gps_frames, ignore_index=True)
        df_raw_gps = df_raw_gps.rename(columns={"session_date": "date"})
        df_raw_gps["date"] = pd.to_datetime(df_raw_gps["date"], errors="coerce").dt.normalize()
        dup_raw_gps = df_raw_gps.duplicated(KEYS).sum()
        print(f"\n--- RAW GPS LOGS (*_gps_summary.csv) ---")
        print(f"Total GPS log rows: {len(df_raw_gps):,}")
        print(f"Unique athletes: {df_raw_gps['athlete_id'].nunique()}")
        print(f"Duplicate athlete-date entries (multiple GPS files on same day): {dup_raw_gps:,}")

    # ─────────────────────────────────────────────────────────────
    # PART 2: SOURCE TABLES (Training, Wellness, GPS, Injuries)
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PART 2: DAILY TABLES AUDIT (Partner's Audit Script)")
    print("=" * 65)

    training = load_training_load()
    wellness = load_wellness()
    gps = load_gps()
    injuries = load_injuries()

    tables = {
        "training": training,
        "wellness": wellness,
        "gps": gps,
    }

    # Normalize dates
    for name, df in tables.items():
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

        print(f"\n--- {name.upper()} ---")
        print("Rows:", f"{len(df):,}")
        print("Athletes:", df["athlete_id"].nunique())
        print("Missing keys:", df[KEYS].isna().any(axis=1).sum())
        dup_count = df.duplicated(KEYS).sum()
        print("Duplicate athlete-days:", f"{dup_count:,}")
        print("Date range:", df["date"].min().strftime("%Y-%m-%d"), "to", df["date"].max().strftime("%Y-%m-%d"))
        print("Median rows per athlete:", df.groupby("athlete_id")["date"].count().median())

    # Check whether join keys are unique before merging
    print("\n--- DUPLICATE KEYS CHECK ---")
    for name, df in tables.items():
        duplicates = df[df.duplicated(KEYS, keep=False)]
        if not duplicates.empty:
            print(f"\nWARNING: {name} has duplicate keys ({df.duplicated(KEYS).sum():,} duplicates):")
            print(duplicates[KEYS].head(10).to_string(index=False))
        else:
            print(f"[OK] {name}: No duplicate athlete-date keys (0 duplicates)")

    # Measure overlap between tables
    t_keys = set(zip(training["athlete_id"], training["date"]))
    w_keys = set(zip(wellness["athlete_id"], wellness["date"]))
    g_keys = set(zip(gps["athlete_id"], gps["date"]))

    print("\n--- OVERLAP & SET INTERSECTIONS ---")
    print(f"Unique Training keys:         {len(t_keys):,}")
    print(f"Unique Wellness keys:         {len(w_keys):,}")
    print(f"Unique GPS keys:              {len(g_keys):,}")
    print(f"Training & Wellness intersection:       {len(t_keys & w_keys):,}")
    print(f"Training & GPS intersection:            {len(t_keys & g_keys):,}")
    print(f"Wellness & GPS intersection:            {len(w_keys & g_keys):,}")
    print(f"All three intersection (INNER JOIN):    {len(t_keys & w_keys & g_keys):,}")
    print(f"All three union (FULL OUTER JOIN):      {len(t_keys | w_keys | g_keys):,}")

    # Measure how many training rows have matching records
    print("\n--- TRAINING MATCHING / MISSING ---")
    for name, df in [
        ("wellness", wellness),
        ("gps", gps),
    ]:
        keys = df[KEYS].drop_duplicates()
        check = training[KEYS].merge(
            keys,
            on=KEYS,
            how="left",
            indicator=True,
            validate="many_to_one"
        )
        matching = (check["_merge"] == "both").sum()
        missing = (check["_merge"] == "left_only").sum()
        print(f"Training rows matching {name}: {matching:,} ({matching / len(training) * 100:.1f}%)")
        print(f"Training rows missing {name}:  {missing:,} ({missing / len(training) * 100:.1f}%)")

    # ─────────────────────────────────────────────────────────────
    # PART 3: INJURIES AUDIT
    # ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PART 3: INJURIES AUDIT")
    print("=" * 65)

    injuries["date"] = pd.to_datetime(injuries["date"], errors="coerce").dt.normalize()
    print("Total injury records:", len(injuries))
    print("Unique injured athletes:", injuries["athlete_id"].nunique())
    inj_dup = injuries.duplicated(subset=KEYS).sum()
    print("Duplicate athlete-injury dates:", inj_dup)
    if inj_dup > 0:
        print("\nDuplicate injury details (athlete reported multiple injuries on the same date):")
        dup_inj = injuries[injuries.duplicated(subset=KEYS, keep=False)]
        print(dup_inj[["athlete_id", "date", "body_area", "severity"]].head(10).to_string(index=False))

    # Match injuries against athlete keys in the dataset
    inj_keys = set(zip(injuries["athlete_id"], injuries["date"]))
    all_keys = t_keys | w_keys | g_keys
    print(f"\nInjuries matching athlete-dates in dataset: {len(inj_keys & all_keys)} / {len(inj_keys)}")

    # ─────────────────────────────────────────────────────────────
    # PART 4: PROCESSED DATASET (daily_observations.csv)
    # ─────────────────────────────────────────────────────────────
    proc_path = OUTPUT_DIR / "daily_observations.csv"
    if proc_path.exists():
        print("\n" + "=" * 65)
        print("PART 4: PROCESSED DATASET (data/processed/daily_observations.csv)")
        print("=" * 65)
        df_proc = pd.read_csv(proc_path)
        df_proc["date"] = pd.to_datetime(df_proc["date"], errors="coerce").dt.normalize()
        print(f"Total rows: {len(df_proc):,}")
        print(f"Athletes: {df_proc['athlete_id'].nunique()}")
        print(f"Duplicate athlete-date keys: {df_proc.duplicated(KEYS).sum():,}")
        print(f"Injury target positive (injury_next_7d=1): {int(df_proc['injury_next_7d'].sum()):,} ({df_proc['injury_next_7d'].mean() * 100:.2f}%)")
        print(f"Injury target negative (injury_next_7d=0): {int((df_proc['injury_next_7d'] == 0).sum()):,}")

    print("\n" + "=" * 65)
    print("   AUDIT COMPLETE")
    print("=" * 65)

if __name__ == "__main__":
    audit()
