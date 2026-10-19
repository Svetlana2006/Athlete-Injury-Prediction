"""
Classical Machine Learning Benchmark & Comparison Pipeline
===========================================================
Trains and benchmarks tree-based and linear models on the same time-aware split:
1. Balanced Logistic Regression (Linear baseline)
2. Balanced Random Forest (Tree Ensemble baseline)
3. Balanced HistGradientBoosting (Gradient Boosting baseline)

Evaluates holdout metrics (ROC-AUC, PR-AUC, Brier score, Top-10% Recall),
extracts sports-science feature importances, and saves comparison artifacts.

Usage:
    python app/ml/train_classical.py
"""

from datetime import datetime
import json
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.ml.dataset import FEATURE_COLUMNS, TARGET_COL

DATA_PATH = BASE_DIR / "data" / "processed" / "daily_observations.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


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


def train_and_evaluate_classical(
    filepath: Path = DATA_PATH,
    split_ratio: float = 0.80,
    split_date: str | None = None,
) -> dict:
    if not filepath.exists():
        raise FileNotFoundError(f"Processed dataset not found at {filepath}")

    print("=" * 66)
    print("   ATHLETE INJURY PREDICTION: CLASSICAL ML BENCHMARK")
    print("=" * 66)

    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["athlete_id", "date"]).reset_index(drop=True)

    available_features = [f for f in FEATURE_COLUMNS if f in df.columns]

    if split_date:
        cutoff = pd.to_datetime(split_date)
    else:
        unique_dates = pd.Series(df["date"].unique()).sort_values()
        cutoff_idx = int(len(unique_dates) * split_ratio)
        cutoff = unique_dates.iloc[cutoff_idx]

    print(f"Time-Aware Cutoff Date  : {cutoff.strftime('%Y-%m-%d')}")
    print(f"Feature Count           : {len(available_features)}")

    # Preprocessing with sports science domain imputation
    train_mask = df["date"] < cutoff
    train_df = df[train_mask].copy()
    test_df = df[~train_mask].copy()

    y_train = train_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values

    baseline_rate = float(y_test.mean())
    print(f"Train Observations      : {len(train_df):,} (Injuries: {y_train.sum():,} = {y_train.mean():.2%})")
    print(f"Test Observations       : {len(test_df):,} (Injuries: {y_test.sum():,} = {baseline_rate:.2%})")

    # Domain fix: uninjured athletes receive 730 days (2 years baseline)
    if "days_since_last_injury" in train_df.columns:
        train_df["days_since_last_injury"] = train_df["days_since_last_injury"].fillna(730.0)
        test_df["days_since_last_injury"] = test_df["days_since_last_injury"].fillna(730.0)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(train_df[available_features])
    X_test_imp = imputer.transform(test_df[available_features])

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    models = {
        "Logistic Regression (Balanced)": {
            "model": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
        },
        "Random Forest (Balanced, max_depth=6)": {
            "model": RandomForestClassifier(
                n_estimators=150,
                max_depth=6,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            ),
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
        },
        "HistGradientBoosting (Balanced)": {
            "model": HistGradientBoostingClassifier(
                class_weight="balanced",
                max_depth=5,
                random_state=42,
            ),
            # Gradient boosting natively handles missing values without scaling
            "X_train": train_df[available_features],
            "X_test": test_df[available_features],
        },
    }

    results = {}
    print("\n" + "=" * 66)
    print("TRAINING & EVALUATING BENCHMARKS")
    print("=" * 66)

    for name, cfg in models.items():
        clf = cfg["model"]
        print(f"Fitting {name} ...")
        clf.fit(cfg["X_train"], y_train)
        probs = clf.predict_proba(cfg["X_test"])[:, 1]
        m = compute_metrics(y_test, probs)
        results[name] = {
            "metrics": m,
            "clf": clf,
        }
        print(f"  -> ROC-AUC: {m['roc_auc']:.4f} | PR-AUC: {m['pr_auc']:.4f} | Top-10% Recall: {m['top10_recall']*100:.1f}%")

    # Feature Importance from Random Forest
    rf_model = results["Random Forest (Balanced, max_depth=6)"]["clf"]
    importances = pd.Series(rf_model.feature_importances_, index=available_features).sort_values(ascending=False)

    print("\n" + "=" * 66)
    print("TOP 10 SPORTS SCIENCE PREDICTORS (RANDOM FOREST)")
    print("=" * 66)
    for feat, imp_val in importances.head(10).items():
        bar = "=" * int(imp_val * 60)
        print(f"  {feat:<24} : {imp_val*100:5.1f}% {bar}")

    # Check for Deep Model results to include in side-by-side comparison
    deep_results_path = ARTIFACTS_DIR / "deep_model_results.json"
    deep_metrics = None
    if deep_results_path.exists():
        try:
            with open(deep_results_path, encoding="utf-8") as f:
                d_data = json.load(f)
                deep_metrics = d_data.get("metrics")
                deep_arch = d_data.get("architecture", "Deep RNN")
        except Exception:
            pass

    # Print Full Comparative Table
    print("\n" + "=" * 66)
    print("MODEL COMPARISON TABLE (UNSEEN HOLDOUT SET)")
    print("=" * 66)
    print(f"{'Model':<36} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Top-10% Recall':<14}")
    print("-" * 72)
    if deep_metrics:
        print(f"{deep_arch:<36} | {deep_metrics['roc_auc']:<8.4f} | {deep_metrics['pr_auc']:<8.4f} | {deep_metrics['top10_recall']*100:<13.1f}%")

    for name, r in results.items():
        m = r["metrics"]
        print(f"{name:<36} | {m['roc_auc']:<8.4f} | {m['pr_auc']:<8.4f} | {m['top10_recall']*100:<13.1f}%")
    print(f"{'[Random Baseline]':<36} | {'0.5000':<8} | {baseline_rate:<8.4f} | {'10.0%':<14}")
    print("=" * 66)

    # Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save winning Random Forest production bundle
    rf_bundle = {
        "model": rf_model,
        "imputer": imputer,
        "scaler": scaler,
        "feature_cols": available_features,
        "metrics": results["Random Forest (Balanced, max_depth=6)"]["metrics"],
        "feature_importances": importances.to_dict(),
        "created_at": datetime.now().isoformat(),
    }
    rf_bundle_path = ARTIFACTS_DIR / "rf_injury_model.joblib"
    joblib.dump(rf_bundle, rf_bundle_path)
    print(f"\nSaved production Random Forest model to {rf_bundle_path}")

    # 2. Save full comparison JSON
    comparison_data = {
        "timestamp": datetime.now().isoformat(),
        "cutoff_date": cutoff.strftime("%Y-%m-%d"),
        "baseline_injury_rate": baseline_rate,
        "models": {
            k: {
                "roc_auc": v["metrics"]["roc_auc"],
                "pr_auc": v["metrics"]["pr_auc"],
                "brier_score": v["metrics"]["brier_score"],
                "top10_recall": v["metrics"]["top10_recall"],
                "top10_precision": v["metrics"]["top10_precision"],
            }
            for k, v in results.items()
        },
        "top_features": importances.head(15).to_dict(),
    }
    if deep_metrics:
        comparison_data["models"][deep_arch] = deep_metrics

    comp_path = ARTIFACTS_DIR / "model_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"Saved benchmark comparison to {comp_path}")

    return comparison_data


if __name__ == "__main__":
    train_and_evaluate_classical()
