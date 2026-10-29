"""
AutoGluon Tabular AutoML Pipeline for Athlete Injury Prediction
==============================================================
Trains and benchmarks ALL major machine learning and deep learning algorithms
in a single unified pass using AWS AutoGluon.

Supports two feature modes:
  --mode engineered  : 43-column Tier-2 feature set (default, best known result)
  --mode raw         : 22-column raw sensor readings only (no derived formulas)

Usage:
    pip install autogluon
    python app/ml/train_autogluon.py --split-date 2021-01-01
    python app/ml/train_autogluon.py --split-date 2021-01-01 --mode raw
    python app/ml/train_autogluon.py --time-limit 600 --presets best_quality --mode raw
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_PATH = BASE_DIR / "data" / "processed" / "daily_observations.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
AG_MODELS_DIR = ARTIFACTS_DIR / "autogluon_models"

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.ml.dataset import FEATURE_COLUMNS, TARGET_COL
from app.ml.nn_benchmark import RAW_FEATURE_COLUMNS


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Computes PR-AUC, ROC-AUC, Brier Score, and Top-10% Triage Recall."""
    pr_auc = float(average_precision_score(y_true, probs))
    roc_auc = float(roc_auc_score(y_true, probs))
    brier = float(brier_score_loss(y_true, probs))

    cutoff = np.percentile(probs, 90)
    preds = (probs >= cutoff).astype(int)

    rec = float(recall_score(y_true, preds, zero_division=0))
    prec = float(precision_score(y_true, preds, zero_division=0))
    f1 = float(f1_score(y_true, preds, zero_division=0))

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "top10_recall": rec,
        "top10_precision": prec,
        "top10_f1": f1,
    }


def train_autogluon(
    filepath: Path = DATA_PATH,
    split_ratio: float = 0.80,
    split_date: str | None = None,
    time_limit: int = 300,
    presets: str = "medium_quality",
    eval_metric: str = "roc_auc",
    mode: str = "engineered",
) -> None:
    try:
        import importlib
        ag_tab = importlib.import_module("autogluon.tabular")
        TabularDataset = getattr(ag_tab, "TabularDataset")
        TabularPredictor = getattr(ag_tab, "TabularPredictor")
    except (ImportError, ModuleNotFoundError):
        print("=" * 66)
        print("ERROR: AutoGluon is not installed.")
        print("Please install AutoGluon by running:")
        print("    pip install autogluon")
        print("=" * 66)
        sys.exit(1)

    if not filepath.exists():
        print(f"Processed dataset not found at {filepath}.")
        print("Automatically triggering data/pipeline.py to build it from raw data...")
        from data.pipeline import main as run_pipeline
        run_pipeline()

    print("=" * 66)
    print("   ATHLETE INJURY PREDICTION: AUTOGLUON AUTOML PIPELINE")
    print("=" * 66)
    print(f"Feature Mode         : {mode.upper()}")
    print(f"AutoML Presets       : {presets}")
    print(f"Target Metric        : {eval_metric}")
    print(f"Time Budget          : {time_limit} seconds ({time_limit/60:.1f} minutes)")

    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["athlete_id", "date"]).reset_index(drop=True)

    # Drop blackout rows (injury_next_7d is NaN — injury day + 2d post-injury).
    n_before = len(df)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Dropped {n_dropped} blackout rows (injury day + 2d post-injury).")

    # Select feature set based on mode
    if mode == "raw":
        candidate_cols = RAW_FEATURE_COLUMNS
    else:
        candidate_cols = FEATURE_COLUMNS
    available_features = [f for f in candidate_cols if f in df.columns]

    if split_date:
        cutoff = pd.to_datetime(split_date)
    else:
        unique_dates = pd.Series(df["date"].unique()).sort_values()
        cutoff_idx = int(len(unique_dates) * split_ratio)
        cutoff = unique_dates.iloc[cutoff_idx]

    print(f"Time-Aware Cutoff Date: {cutoff.strftime('%Y-%m-%d')}")
    print(f"Features Included     : {len(available_features)}")

    train_mask = df["date"] < cutoff
    train_df = df[train_mask][available_features + [TARGET_COL]].copy()
    test_df = df[~train_mask][available_features + [TARGET_COL]].copy()

    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values
    baseline_rate = float(y_test.mean())

    print(f"Train Observations    : {len(train_df):,} (Injuries: {y_train.sum():,} = {y_train.mean():.2%})")
    print(f"Test Observations     : {len(test_df):,} (Injuries: {y_test.sum():,} = {baseline_rate:.2%})")

    # Train TabularPredictor across all algorithms
    print("\n" + "=" * 66)
    print("LAUNCHING AUTOGLUON TRAINING (FITTING ALL MODELS)")
    print("=" * 66)

    predictor = TabularPredictor(
        label=TARGET_COL,
        eval_metric=eval_metric,
        problem_type="binary",
        path=str(AG_MODELS_DIR),
    ).fit(
        train_data=train_df,
        presets=presets,
        time_limit=time_limit,
    )

    # Leaderboard on Unseen Holdout Test Data
    print("\n" + "=" * 66)
    print("AUTOGLUON MODEL LEADERBOARD (UNSEEN HOLDOUT SET)")
    print("=" * 66)
    leaderboard = predictor.leaderboard(test_df, silent=False)

    # Evaluate best predictor
    probs = predictor.predict_proba(test_df.drop(columns=[TARGET_COL]))[1].values
    best_metrics = compute_metrics(y_test, probs)
    best_model_name = predictor.model_best

    print("\n" + "=" * 66)
    print(f"WINNING MODEL: {best_model_name}")
    print("=" * 66)
    print(f"ROC-AUC               : {best_metrics['roc_auc']:.4f}")
    print(f"PR-AUC                : {best_metrics['pr_auc']:.4f} (Baseline: {baseline_rate:.4f})")
    print(f"Brier Score           : {best_metrics['brier_score']:.4f}")
    print(f"Top-10% Triage Recall : {best_metrics['top10_recall']*100:.1f}%")
    print(f"Top-10% Precision     : {best_metrics['top10_precision']*100:.2f}%")

    # Feature Importance
    try:
        print("\n" + "=" * 66)
        print("FEATURE IMPORTANCE (WINNING MODEL)")
        print("=" * 66)
        fi = predictor.feature_importance(test_df, silent=True)
        for feat, score in fi["importance"].head(10).items():
            print(f"  {feat:<25} : {score:+.4f}")
    except Exception as e:
        print(f"Could not compute feature importance: {e}")

    # Save summary artifact — separate file per mode so both results are preserved
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "timestamp": datetime.now().isoformat(),
        "feature_mode": mode,
        "winning_model": best_model_name,
        "metrics": best_metrics,
        "baseline_injury_rate": baseline_rate,
        "leaderboard": leaderboard[["model", "score_test", "pred_time_test"]].head(10).to_dict(orient="records"),
    }
    filename = f"autogluon_results_{mode}.json"
    out_path = ARTIFACTS_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved AutoGluon results to {out_path}")

    # Also update model_comparison.json
    comp_path = ARTIFACTS_DIR / "model_comparison.json"
    if comp_path.exists():
        try:
            with open(comp_path, "r", encoding="utf-8") as f:
                comp_data = json.load(f)
            comp_data["models"][f"AutoGluon ({best_model_name})"] = best_metrics
            with open(comp_path, "w", encoding="utf-8") as f:
                json.dump(comp_data, f, indent=2)
            print(f"Updated {comp_path} with AutoGluon benchmark")
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoGluon Tabular AutoML Pipeline")
    parser.add_argument("--time-limit", type=int, default=300, help="Training time limit in seconds (default: 300)")
    parser.add_argument("--presets", type=str, default="medium_quality", choices=["medium_quality", "high_quality", "best_quality"], help="AutoGluon preset")
    parser.add_argument("--eval-metric", type=str, default="roc_auc", help="Evaluation metric (default: roc_auc)")
    parser.add_argument("--split-date", type=str, default=None, help="Explicit train/test cutoff date YYYY-MM-DD")
    parser.add_argument("--split-ratio", type=float, default=0.80, help="Train/test temporal split ratio if no --split-date given")
    parser.add_argument("--mode", type=str, default="engineered", choices=["raw", "engineered"],
                        help="Feature set: 'raw' (22 sensor columns) or 'engineered' (43 Tier-2 columns, default)")
    args = parser.parse_args()

    train_autogluon(
        time_limit=args.time_limit,
        presets=args.presets,
        eval_metric=args.eval_metric,
        split_date=args.split_date,
        split_ratio=args.split_ratio,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
