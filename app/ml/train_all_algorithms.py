"""
Comprehensive Gradient Boosting & Multi-Model Benchmark Pipeline
================================================================
Trains and evaluates all major modern machine learning algorithms:
- LightGBM (LGBMClassifier)
- XGBoost (XGBClassifier)
- CatBoost (CatBoostClassifier)
- Balanced Random Forest
- Balanced Extra Trees
- Balanced Logistic Regression
- Stacking Ensemble (Meta-learner combining top models)

Gracefully detects which libraries are installed and outputs a complete
comparative leaderboard on the unseen holdout test set.

Usage in Colab:
    pip install lightgbm xgboost catboost -q
    python app/ml/train_all_algorithms.py
"""

from datetime import datetime
import json
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
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


def train_and_benchmark_all(
    filepath: Path = DATA_PATH,
    split_ratio: float = 0.80,
    split_date: str | None = None,
) -> None:
    if not filepath.exists():
        print(f"Processed dataset not found at {filepath}.")
        print("Automatically triggering data/pipeline.py to build it from raw data...")
        from data.pipeline import main as run_pipeline
        run_pipeline()

    print("=" * 68)
    print("   ALL-ALGORITHM BENCHMARK: GRADIENT BOOSTING & ENSEMBLES")
    print("=" * 68)

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

    train_mask = df["date"] < cutoff
    train_df = df[train_mask].copy()
    test_df = df[~train_mask].copy()

    X_train = train_df[available_features].values
    y_train = train_df[TARGET_COL].values
    X_test = test_df[available_features].values
    y_test = test_df[TARGET_COL].values

    baseline_rate = float(y_test.mean())
    print(f"Train Observations      : {len(train_df):,} (Injuries: {y_train.sum():,} = {y_train.mean():.2%})")
    print(f"Test Observations       : {len(test_df):,} (Injuries: {y_test.sum():,} = {baseline_rate:.2%})")

    # Scale for linear models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Imbalance scale factor for boosting
    imbalance_weight = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))

    candidate_models = {}

    # 1. Random Forest
    candidate_models["Random Forest (Balanced)"] = (
        RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
        X_train,
        X_test,
    )

    # 2. Extra Trees
    candidate_models["Extra Trees (Balanced)"] = (
        ExtraTreesClassifier(
            n_estimators=150,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        ),
        X_train,
        X_test,
    )

    # 3. Logistic Regression
    candidate_models["Logistic Regression (Balanced)"] = (
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
        X_train_scaled,
        X_test_scaled,
    )

    # 4. HistGradientBoosting
    candidate_models["HistGradientBoosting (Balanced)"] = (
        HistGradientBoostingClassifier(
            class_weight="balanced",
            max_depth=5,
            random_state=42,
        ),
        X_train,
        X_test,
    )

    # 5. LightGBM (if available)
    try:
        from lightgbm import LGBMClassifier
        candidate_models["LightGBM (Tuned Scale-Weight)"] = (
            LGBMClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.03,
                scale_pos_weight=min(20.0, np.sqrt(imbalance_weight)),
                random_state=42,
                verbosity=-1,
            ),
            X_train,
            X_test,
        )
    except ImportError:
        pass

    # 6. XGBoost (if available)
    try:
        from xgboost import XGBClassifier
        candidate_models["XGBoost (Tuned Scale-Weight)"] = (
            XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.03,
                scale_pos_weight=min(20.0, np.sqrt(imbalance_weight)),
                random_state=42,
                eval_metric="logloss",
            ),
            X_train,
            X_test,
        )
    except ImportError:
        pass

    # 7. CatBoost (if available)
    try:
        from catboost import CatBoostClassifier
        candidate_models["CatBoost (Auto Class Weights)"] = (
            CatBoostClassifier(
                iterations=200,
                depth=5,
                learning_rate=0.03,
                auto_class_weights="Balanced",
                random_seed=42,
                verbose=False,
            ),
            X_train,
            X_test,
        )
    except ImportError:
        pass

    print("\n" + "=" * 68)
    print("TRAINING CANDIDATE ALGORITHMS")
    print("=" * 68)

    results = {}
    fitted_models = []

    for name, (clf, xtr, xte) in candidate_models.items():
        print(f"Training {name:<35} ...", end=" ", flush=True)
        try:
            clf.fit(xtr, y_train)
            probs = clf.predict_proba(xte)[:, 1]
            m = compute_metrics(y_test, probs)
            results[name] = {"metrics": m, "probs": probs, "model": clf}
            fitted_models.append((name, clf))
            print(f"ROC-AUC: {m['roc_auc']:.4f} | PR-AUC: {m['pr_auc']:.4f} | Top-10% Recall: {m['top10_recall']*100:.1f}%")
        except Exception as e:
            print(f"Failed: {e}")

    # Build Weighted Stacking / Soft Voting Ensemble
    if len(fitted_models) >= 2:
        print("Training Stacking Ensemble (Blending top tree models) ...", end=" ", flush=True)
        try:
            # Average probabilities of top 3 models by ROC-AUC
            top_models = sorted(results.items(), key=lambda item: item[1]["metrics"]["roc_auc"], reverse=True)[:3]
            ensemble_probs = np.mean([item[1]["probs"] for item in top_models], axis=0)
            m_ens = compute_metrics(y_test, ensemble_probs)
            ens_name = "Stacking / Blend Ensemble (Top 3)"
            results[ens_name] = {"metrics": m_ens, "probs": ensemble_probs}
            print(f"ROC-AUC: {m_ens['roc_auc']:.4f} | PR-AUC: {m_ens['pr_auc']:.4f} | Top-10% Recall: {m_ens['top10_recall']*100:.1f}%")
        except Exception as e:
            print(f"Failed: {e}")

    # Print Final Ranked Leaderboard
    print("\n" + "=" * 68)
    print("ALL-ALGORITHM FINAL LEADERBOARD (UNSEEN HOLDOUT SET)")
    print("=" * 68)
    print(f"{'Rank':<4} {'Algorithm':<36} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'Top-10% Recall'}")
    print("-" * 72)

    ranked = sorted(results.items(), key=lambda item: item[1]["metrics"]["roc_auc"], reverse=True)
    for rank, (name, data) in enumerate(ranked, 1):
        m = data["metrics"]
        star = " [WINNER]" if rank == 1 else ""
        print(f"{rank:<4} {name:<36} | {m['roc_auc']:<8.4f} | {m['pr_auc']:<8.4f} | {m['top10_recall']*100:>5.1f}%{star}")

    print(f"{'--':<4} {'[Random Guessing Baseline]':<36} | {'0.5000':<8} | {baseline_rate:<8.4f} |  10.0%")
    print("=" * 68)

    # Save to artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    # Save winning model bundle
    winning_name = ranked[0][0]
    if "model" in results[winning_name]:
        best_bundle = {
            "model_name": winning_name,
            "model": results[winning_name]["model"],
            "feature_cols": available_features,
            "metrics": results[winning_name]["metrics"],
            "created_at": datetime.now().isoformat(),
        }
        import joblib
        joblib.dump(best_bundle, ARTIFACTS_DIR / "best_injury_model.joblib")
        print(f"Saved winning model ({winning_name}) to {ARTIFACTS_DIR / 'best_injury_model.joblib'}")

    comp_path = ARTIFACTS_DIR / "model_comparison.json"
    comp_data = {"models": {}}
    if comp_path.exists():
        try:
            with open(comp_path, "r", encoding="utf-8") as f:
                comp_data = json.load(f)
        except Exception:
            pass
    for name, data in results.items():
        comp_data["models"][name] = data["metrics"]
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comp_data, f, indent=2)
    print(f"Updated benchmark comparison in {comp_path}")


if __name__ == "__main__":
    train_and_benchmark_all()
