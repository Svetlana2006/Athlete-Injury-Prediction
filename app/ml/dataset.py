"""
Sequential Dataset Builder for Athlete Injury Prediction
========================================================
Implements Flowchart Steps:
1. Load cleaned daily player records
2. Build Historical Sequences for Each Athlete (sliding lookback window L, e.g. 14 days)
3. Strict Time-Aware Train / Test Split (no future leakage)
4. Imputation and Normalization (fitted strictly on training sequences)
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "daily_observations.csv"

# Daily Multimodal Features across Workload, Wellness, GPS, and History
FEATURE_COLUMNS = [
    # Workload / Training Load
    "daily_load",
    "acwr",
    "atl",
    "ctl28",
    "ctl42",
    "monotony",
    "strain",
    "weekly_load",
    "rpe",
    "duration_minutes",
    "session_rpe",
    "days_since_rest",
    # Tier 2: ACWR danger zone & spike
    "acwr_danger",        # binary: ACWR > 1.5 (Gabbett red zone)
    "acwr_spike",         # week-on-week ACWR change (sudden jump signal)
    # Tier 2: Consecutive load streak
    "high_load_streak",   # consecutive days above athlete's personal median load
    # Wellness
    "fatigue",
    "soreness",
    "mood",
    "readiness",
    "sleep_duration",
    "sleep_quality",
    "stress",
    "composite_wellness",
    "sleep_deficit",
    "fatigue_trend_7d",
    "soreness_trend_7d",
    "wellness_delta",
    # GPS / Movement
    "total_distance_m",
    "high_speed_distance_m",
    "sprint_distance_m",
    "max_speed_m_s",
    "mean_speed_m_s",
    "hsr_ratio",
    # Clinical & Injury History
    "prior_injury_count",
    "days_since_last_injury",
    "has_prior_injury",
    "illness_last_7d",
    "is_rest_day",
    "has_gps",
    # Tier 2: Athlete-relative z-scores (deviation from own baseline)
    "fatigue_athlete_z",  # fatigue relative to this athlete's own mean/std
    "soreness_athlete_z", # soreness relative to this athlete's own mean/std
    "daily_load_athlete_z",  # load relative to this athlete's own mean/std
    "acwr_athlete_z",        # ACWR relative to this athlete's own mean/std
]

TARGET_COL = "injury_next_7d"


class AthleteSequenceDataset(Dataset):
    """PyTorch Dataset of historical athlete sequences."""

    def __init__(
        self,
        sequences: np.ndarray,
        targets: np.ndarray,
        athlete_ids: list[str],
        dates: list[str],
    ):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.athlete_ids = athlete_ids
        self.dates = dates

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        return self.sequences[idx], self.targets[idx], self.athlete_ids[idx], self.dates[idx]


def build_athlete_sequences(
    df: pd.DataFrame,
    seq_len: int = 14,
    feature_cols: list[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Builds historical sliding sequences of length `seq_len` for each athlete.
    For day t, input sequence is [t - seq_len + 1, ..., t], and target is injury_next_7d at t.
    """
    if feature_cols is None:
        feature_cols = [f for f in FEATURE_COLUMNS if f in df.columns]

    seq_list = []
    target_list = []
    athlete_list = []
    date_list = []

    # Sort strictly by athlete and date
    df_sorted = df.sort_values(["athlete_id", "date"]).reset_index(drop=True)

    for athlete_id, grp in df_sorted.groupby("athlete_id"):
        grp = grp.reset_index(drop=True)
        if len(grp) < seq_len:
            continue

        feat_vals = grp[feature_cols].values
        target_vals = grp[TARGET_COL].values
        date_vals = grp["date"].dt.strftime("%Y-%m-%d").values

        for i in range(seq_len - 1, len(grp)):
            seq = feat_vals[i - seq_len + 1 : i + 1]  # shape (seq_len, num_features)
            target = target_vals[i]
            target_date = date_vals[i]

            seq_list.append(seq)
            target_list.append(target)
            athlete_list.append(athlete_id)
            date_list.append(target_date)

    return (
        np.array(seq_list, dtype=np.float32),
        np.array(target_list, dtype=np.float32),
        athlete_list,
        date_list,
    )


def prepare_sequential_data(
    filepath: Path = DATA_PATH,
    seq_len: int = 14,
    split_ratio: float = 0.80,
    split_date: str | None = None,
) -> dict:
    """End-to-end data preparation:
    - Loads SoccerMon dataset
    - Fits Imputer and Scaler on TRAINING rows only
    - Transforms full dataset with train-fitted preprocessors
    - Builds sequences on the FULL per-athlete timeline (critical: test
      sequences get real historical lookback into the training period)
    - Splits train / test sequences by prediction date (zero leakage)
    - Returns PyTorch datasets and preprocessing metadata
    """
    if not filepath.exists():
        print(f"Processed dataset not found at {filepath}.")
        print("Automatically triggering data/pipeline.py to build it from raw data...")
        from data.pipeline import main as run_pipeline
        run_pipeline()

    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["athlete_id", "date"]).reset_index(drop=True)

    # Drop blackout rows (injury_next_7d == NaN).
    # These are the injury date and 2 days post-injury; their features are
    # contaminated by forward-fill from post-injury sessions.
    n_before = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} blackout rows (injury day + 2d post-injury).")

    available_features = [f for f in FEATURE_COLUMNS if f in df.columns]

    # Time-aware split date
    if split_date:
        cutoff = pd.to_datetime(split_date)
    else:
        unique_dates = pd.Series(df["date"].unique()).sort_values()
        cutoff_idx = int(len(unique_dates) * split_ratio)
        cutoff = unique_dates.iloc[cutoff_idx]

    # Fit imputer and scaler on training rows only — never touch future
    train_mask = df["date"] < cutoff
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    df_features = df[available_features].astype(np.float64).copy()

    # Domain imputation:
    # For athletes with 0 prior injuries, days_since_last_injury is NaN.
    # Filling with median falsely marks them as having an injury ~45 days ago.
    # Fill with 730.0 (2 years = uninjured baseline) to preserve clean signal.
    if "days_since_last_injury" in df_features.columns:
        df_features["days_since_last_injury"] = df_features["days_since_last_injury"].fillna(730.0)

    train_indices = df_features.index[train_mask]
    test_indices = df_features.index[~train_mask]

    train_imputed = imputer.fit_transform(df_features.loc[train_indices])
    test_imputed = imputer.transform(df_features.loc[test_indices])

    train_scaled = scaler.fit_transform(train_imputed)
    test_scaled = scaler.transform(test_imputed)

    df_features.loc[train_indices] = train_scaled
    df_features.loc[test_indices] = test_scaled
    df[available_features] = df_features

    # Build sequences across the FULL per-athlete timeline
    # The split is done on the PREDICTION DATE — not the row date.
    # This ensures every test sequence has a real 14-day historical window.
    all_X, all_y, all_athletes, all_dates = build_athlete_sequences(
        df, seq_len=seq_len, feature_cols=available_features
    )
    all_dates_dt = pd.to_datetime(all_dates)

    train_mask_seq = all_dates_dt < cutoff
    test_mask_seq = all_dates_dt >= cutoff

    X_train = all_X[train_mask_seq]
    y_train = all_y[train_mask_seq]
    train_athletes = [a for a, m in zip(all_athletes, train_mask_seq) if m]
    train_dates = [d for d, m in zip(all_dates, train_mask_seq) if m]

    X_test = all_X[test_mask_seq]
    y_test = all_y[test_mask_seq]
    test_athletes = [a for a, m in zip(all_athletes, test_mask_seq) if m]
    test_dates = [d for d, m in zip(all_dates, test_mask_seq) if m]

    train_dataset = AthleteSequenceDataset(X_train, y_train, train_athletes, train_dates)
    test_dataset = AthleteSequenceDataset(X_test, y_test, test_athletes, test_dates)

    print("\n" + "=" * 62)
    print("SEQUENTIAL TIME-AWARE DATASET READY")
    print("=" * 62)
    print(f"Lookback window (Sequence Length) : {seq_len} days")
    print(f"Multimodal Features per step     : {len(available_features)}")
    print(f"Cutoff Date (Train / Test)       : {cutoff.date()}")
    print(f"Train Sequences                  : {len(train_dataset):,} (Injuries: {int(y_train.sum()):,} = {y_train.mean()*100:.2f}%)")
    print(f"Test Sequences                   : {len(test_dataset):,} (Injuries: {int(y_test.sum()):,} = {y_test.mean()*100:.2f}%)")

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "feature_cols": available_features,
        "imputer": imputer,
        "scaler": scaler,
        "cutoff_date": str(cutoff.date()),
        "seq_len": seq_len,
    }
