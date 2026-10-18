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
    "illness_last_7d",
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
    - Time-aware chronological train / test split
    - Fits Imputer and Scaler ONLY on train data
    - Builds sequences of shape (N, seq_len, num_features)
    - Returns PyTorch datasets and preprocessing metadata
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Processed dataset not found at {filepath}")

    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    available_features = [f for f in FEATURE_COLUMNS if f in df.columns]

    # Time-aware split date
    if split_date:
        cutoff = pd.to_datetime(split_date)
    else:
        unique_dates = pd.Series(df["date"].unique()).sort_values()
        cutoff_idx = int(len(unique_dates) * split_ratio)
        cutoff = unique_dates.iloc[cutoff_idx]

    train_df = df[df["date"] < cutoff].copy()
    test_df = df[df["date"] >= cutoff].copy()

    # Preprocessors fitted ONLY on training data
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train_df[available_features] = imputer.fit_transform(train_df[available_features])
    train_df[available_features] = scaler.fit_transform(train_df[available_features])

    # Transform test set with fitted train params
    test_df[available_features] = imputer.transform(test_df[available_features])
    test_df[available_features] = scaler.transform(test_df[available_features])

    # Build sequences
    X_train, y_train, train_athletes, train_dates = build_athlete_sequences(
        train_df, seq_len=seq_len, feature_cols=available_features
    )
    X_test, y_test, test_athletes, test_dates = build_athlete_sequences(
        test_df, seq_len=seq_len, feature_cols=available_features
    )

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
