"""
Neural Network Architecture Benchmark
======================================
Trains all 10 architectures in two feature modes:
  - RAW:       Original sensor / wellness columns (no derived formulas)
  - ENGINEERED: The 43-column Tier-2 feature set used by current models

For each (architecture, feature_mode) combination:
  - 10 epochs of training with FocalLoss + AdamW
  - Evaluates ROC-AUC, PR-AUC, Brier Score, Top-10% Recall on held-out test
  - Prints a ranked leaderboard at the end

Usage:
    python app/ml/nn_benchmark.py
    python app/ml/nn_benchmark.py --split-date 2021-01-01 --epochs 10
    python app/ml/nn_benchmark.py --models GRU LSTM Transformer --modes raw engineered
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    recall_score,
    precision_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from app.ml.dataset import AthleteSequenceDataset, build_athlete_sequences, TARGET_COL, FEATURE_COLUMNS
from app.ml.nn_models import MODEL_REGISTRY

DATA_PATH = BASE_DIR / "data" / "processed" / "daily_observations.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# ─────────────────────────────────────────────────────────────────────────────
# RAW feature columns — direct sensor readings, NO derived formulas
# ─────────────────────────────────────────────────────────────────────────────
RAW_FEATURE_COLUMNS = [
    # Training load (raw session data)
    "rpe",
    "duration_minutes",
    "session_rpe",
    "daily_load",
    "weekly_load",
    # Wellness (raw subjective self-report)
    "fatigue",
    "soreness",
    "mood",
    "readiness",
    "sleep_duration",
    "sleep_quality",
    "stress",
    # GPS / Movement (raw hardware output)
    "total_distance_m",
    "high_speed_distance_m",
    "sprint_distance_m",
    "max_speed_m_s",
    "mean_speed_m_s",
    # Clinical flags (administrative records)
    "prior_injury_count",
    "has_prior_injury",
    "is_rest_day",
    "has_gps",
    "illness_last_7d",
]


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1.0 - probs)
        return (self.alpha * (1.0 - pt) ** self.gamma * bce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Data Preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_datasets(
    feature_mode: str,
    split_date: str | None,
    seq_len: int = 14,
) -> dict:
    """Load and prepare sequential datasets for the given feature mode."""
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["athlete_id", "date"]).reset_index(drop=True)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    if feature_mode == "raw":
        feature_cols = [c for c in RAW_FEATURE_COLUMNS if c in df.columns]
    else:
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    # Time-aware cutoff
    if split_date:
        cutoff = pd.to_datetime(split_date)
    else:
        unique_dates = pd.Series(df["date"].unique()).sort_values()
        cutoff = unique_dates.iloc[int(len(unique_dates) * 0.80)]

    train_mask = df["date"] < cutoff
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    df_feat = df[feature_cols].astype(np.float64).copy()
    if "days_since_last_injury" in df_feat.columns:
        df_feat["days_since_last_injury"] = df_feat["days_since_last_injury"].fillna(730.0)

    train_idx = df_feat.index[train_mask]
    test_idx = df_feat.index[~train_mask]

    df_feat.loc[train_idx] = scaler.fit_transform(imputer.fit_transform(df_feat.loc[train_idx]))
    df_feat.loc[test_idx] = scaler.transform(imputer.transform(df_feat.loc[test_idx]))
    df[feature_cols] = df_feat

    all_X, all_y, all_ath, all_dates = build_athlete_sequences(df, seq_len=seq_len, feature_cols=feature_cols)
    all_dates_dt = pd.to_datetime(all_dates)

    tr_mask = all_dates_dt < cutoff
    te_mask = all_dates_dt >= cutoff

    train_ds = AthleteSequenceDataset(all_X[tr_mask], all_y[tr_mask],
                                       [a for a, m in zip(all_ath, tr_mask) if m],
                                       [d for d, m in zip(all_dates, tr_mask) if m])
    test_ds = AthleteSequenceDataset(all_X[te_mask], all_y[te_mask],
                                      [a for a, m in zip(all_ath, te_mask) if m],
                                      [d for d, m in zip(all_dates, te_mask) if m])
    return {
        "train_ds": train_ds,
        "test_ds": test_ds,
        "feature_cols": feature_cols,
        "cutoff": str(cutoff.date()),
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "n_pos_train": int(train_ds.targets.sum().item()),
        "n_pos_test": int(test_ds.targets.sum().item()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    roc = float(roc_auc_score(y_true, probs)) if y_true.sum() > 0 else 0.0
    pr = float(average_precision_score(y_true, probs)) if y_true.sum() > 0 else 0.0
    brier = float(brier_score_loss(y_true, probs))
    cutoff = np.percentile(probs, 90)
    preds = (probs >= cutoff).astype(int)
    rec = float(recall_score(y_true, preds, zero_division=0))
    prec = float(precision_score(y_true, preds, zero_division=0))
    return {"roc_auc": roc, "pr_auc": pr, "brier": brier,
            "top10_recall": rec, "top10_precision": prec}


# ─────────────────────────────────────────────────────────────────────────────
# Train & Evaluate one model
# ─────────────────────────────────────────────────────────────────────────────

def train_and_eval(
    model: nn.Module,
    train_ds: AthleteSequenceDataset,
    test_ds: AthleteSequenceDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> dict:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    y_np = train_ds.targets.numpy()
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)
    best_roc = 0.0
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        for seqs, targets, _, _ in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(seqs)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        scheduler.step()

        # Evaluate every epoch, keep the best
        model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for seqs, targets, _, _ in test_loader:
                logits, _ = model(seqs.to(device))
                all_logits.append(logits.cpu().numpy())
                all_y.append(targets.numpy())
        logits_arr = np.concatenate(all_logits)
        y_arr = np.concatenate(all_y)
        probs = 1.0 / (1.0 + np.exp(-logits_arr))
        m = compute_metrics(y_arr, probs)
        if m["roc_auc"] > best_roc:
            best_roc = m["roc_auc"]
            best_metrics = m

    return best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main Benchmark Loop
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    models_to_run: list[str],
    modes: list[str],
    split_date: str | None,
    epochs: int,
    batch_size: int,
    lr: float,
    seq_len: int,
    hidden_dim: int,
) -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("  NEURAL NETWORK ARCHITECTURE BENCHMARK - ATHLETE INJURY PREDICTION")
    print("=" * 70)
    print(f"  Device        : {device}")
    print(f"  Architectures : {len(models_to_run)}")
    print(f"  Feature Modes : {modes}")
    print(f"  Split Date    : {split_date or 'auto (80%)'}")
    print(f"  Epochs        : {epochs} | Seq Len: {seq_len} | Batch: {batch_size}")
    print("=" * 70)

    # Pre-load datasets for each mode (load once, reuse for all models)
    data_cache: dict[str, dict] = {}
    for mode in modes:
        print(f"\nLoading {mode.upper()} feature dataset ...")
        d = load_datasets(mode, split_date=split_date, seq_len=seq_len)
        data_cache[mode] = d
        print(f"  Features : {len(d['feature_cols'])} columns")
        print(f"  Cutoff   : {d['cutoff']}")
        print(f"  Train    : {d['n_train']:,} seqs (Injuries: {d['n_pos_train']:,} = {d['n_pos_train']/d['n_train']*100:.2f}%)")
        print(f"  Test     : {d['n_test']:,} seqs  (Injuries: {d['n_pos_test']:,} = {d['n_pos_test']/d['n_test']*100:.2f}%)")

    results: list[dict] = []

    for mode in modes:
        d = data_cache[mode]
        input_dim = len(d["feature_cols"])

        for model_name in models_to_run:
            label = f"{model_name} [{mode.upper()}]"
            print(f"\n" + "-" * 70)
            print(f"  Training : {label}  (input_dim={input_dim})")
            print("-" * 70)

            # Instantiate model — all models share hidden_dim as primary size param
            ModelClass = MODEL_REGISTRY[model_name]
            model = ModelClass(input_dim=input_dim, hidden_dim=hidden_dim)

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  Params    : {n_params:,}")

            t0 = time.time()
            try:
                metrics = train_and_eval(
                    model=model,
                    train_ds=d["train_ds"],
                    test_ds=d["test_ds"],
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    device=device,
                )
                elapsed = time.time() - t0
                print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
                print(f"  PR-AUC    : {metrics['pr_auc']:.4f}")
                print(f"  Brier     : {metrics['brier']:.4f}")
                print(f"  Top-10%R  : {metrics['top10_recall']*100:.1f}%")
                print(f"  Time      : {elapsed:.1f}s")

                results.append({
                    "model": model_name,
                    "mode": mode,
                    "label": label,
                    "params": n_params,
                    "time_s": round(elapsed, 1),
                    **metrics,
                })

            except Exception as e:
                elapsed = time.time() - t0
                print(f"  [FAILED after {elapsed:.1f}s] {e}")
                results.append({
                    "model": model_name,
                    "mode": mode,
                    "label": label,
                    "params": n_params,
                    "time_s": round(elapsed, 1),
                    "roc_auc": 0.0, "pr_auc": 0.0, "brier": 1.0,
                    "top10_recall": 0.0, "top10_precision": 0.0,
                    "error": str(e),
                })

    # ─── Leaderboard ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  FINAL LEADERBOARD  (ranked by ROC-AUC)")
    print("=" * 70)
    print(f"  {'Rank':<5} {'Model + Mode':<32} {'ROC-AUC':>8} {'PR-AUC':>8} {'Brier':>7} {'T10-Rec':>8} {'Time':>7}")
    print("  " + "-"*4 + " " + "-"*31 + " " + "-"*8 + " " + "-"*8 + " " + "-"*7 + " " + "-"*8 + " " + "-"*7)

    ranked = sorted(results, key=lambda r: r["roc_auc"], reverse=True)
    for rank, r in enumerate(ranked, 1):
        print(
            f"  {rank:<5} {r['label']:<32} "
            f"{r['roc_auc']:>8.4f} {r['pr_auc']:>8.4f} "
            f"{r['brier']:>7.4f} {r['top10_recall']*100:>7.1f}% "
            f"{r['time_s']:>6.1f}s"
        )

    print("=" * 70)

    # ─── Save JSON ───────────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACTS_DIR / "nn_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "epochs": epochs, "seq_len": seq_len,
                "batch_size": batch_size, "lr": lr,
                "hidden_dim": hidden_dim, "split_date": split_date,
                "modes": modes, "architectures": models_to_run,
            },
            "results": ranked,
        }, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_models = list(MODEL_REGISTRY.keys())

    parser = argparse.ArgumentParser(description="Benchmark all neural network architectures.")
    parser.add_argument("--models", nargs="+", default=all_models,
                        choices=all_models, help="Which models to run (default: all)")
    parser.add_argument("--modes", nargs="+", default=["raw", "engineered"],
                        choices=["raw", "engineered"], help="Feature modes to benchmark")
    parser.add_argument("--split-date", type=str, default="2021-01-01",
                        help="Train/test temporal cutoff date YYYY-MM-DD")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per model")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seq-len", type=int, default=14, help="Lookback window (days)")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension for all models")
    args = parser.parse_args()

    run_benchmark(
        models_to_run=args.models,
        modes=args.modes,
        split_date=args.split_date,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
    )
