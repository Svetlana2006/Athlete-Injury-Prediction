"""
train.py - Transformer Training Framework
==========================================
Proper full-epoch training loop for the Transformer model on raw features.
Framework for further optimisation: hyper-params, regularisation, data augmentation, etc.

Usage
-----
    python app/ml/train.py
    python app/ml/train.py --epochs 100 --lr 3e-4 --hidden-dim 128
    python app/ml/train.py --help

Output (artifacts/runs/<timestamp>/)
    best_model.pt   - weights at best validation ROC-AUC epoch
    train_log.csv   - per-epoch metrics
    results.json    - final test-set evaluation
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from datetime import datetime
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
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from app.ml.dataset import AthleteSequenceDataset, build_athlete_sequences, TARGET_COL
from app.ml.nn_models import TransformerEncoderModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = BASE_DIR / "data" / "processed" / "daily_observations.csv"
RUNS_DIR = BASE_DIR / "artifacts" / "runs"

# Raw sensor columns (22) - no derived formulas, best feature mode from benchmark
RAW_FEATURE_COLUMNS: list[str] = [
    "rpe", "duration_minutes", "session_rpe", "daily_load", "weekly_load",
    "fatigue", "soreness", "mood", "readiness",
    "sleep_duration", "sleep_quality", "stress",
    "total_distance_m", "high_speed_distance_m", "sprint_distance_m",
    "max_speed_m_s", "mean_speed_m_s",
    "prior_injury_count", "has_prior_injury", "is_rest_day", "has_gps", "illness_last_7d",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(split_date: str, seq_len: int, batch_size: int, seed: int) -> dict[str, Any]:
    """
    Loads processed CSV, builds sliding athlete sequences, time-splits at
    split_date, fits imputer + scaler on training rows only (no leakage).
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at {DATA_PATH}.\n"
            "Run data/pipeline.py first to build daily_observations.csv."
        )

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["athlete_id", "date"]).reset_index(drop=True)
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    feature_cols = [c for c in RAW_FEATURE_COLUMNS if c in df.columns]
    cutoff = pd.to_datetime(split_date)

    # Fit imputer / scaler on train rows only
    train_mask = df["date"] < cutoff
    df_feat = df[feature_cols].astype(np.float64).copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train_idx = df_feat.index[train_mask]
    test_idx = df_feat.index[~train_mask]

    df_feat.loc[train_idx] = scaler.fit_transform(imputer.fit_transform(df_feat.loc[train_idx]))
    df_feat.loc[test_idx] = scaler.transform(imputer.transform(df_feat.loc[test_idx]))
    df[feature_cols] = df_feat

    # Build sequences over full per-athlete timeline, split by prediction date
    all_X, all_y, all_athletes, all_dates = build_athlete_sequences(
        df, seq_len=seq_len, feature_cols=feature_cols
    )
    all_dates_dt = pd.to_datetime(all_dates)

    train_mask_seq = all_dates_dt < cutoff
    test_mask_seq = all_dates_dt >= cutoff

    X_train, y_train = all_X[train_mask_seq], all_y[train_mask_seq]
    X_test, y_test = all_X[test_mask_seq], all_y[test_mask_seq]

    train_ds = AthleteSequenceDataset(
        X_train, y_train,
        [a for a, m in zip(all_athletes, train_mask_seq) if m],
        [d for d, m in zip(all_dates, train_mask_seq) if m],
    )
    test_ds = AthleteSequenceDataset(
        X_test, y_test,
        [a for a, m in zip(all_athletes, test_mask_seq) if m],
        [d for d, m in zip(all_dates, test_mask_seq) if m],
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=g)
    val_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                            num_workers=0)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    print(f"  Features   : {len(feature_cols)} raw columns")
    print(f"  Cutoff     : {cutoff.date()}")
    print(f"  Train      : {len(y_train):,}  ({n_pos} injuries = {y_train.mean()*100:.2f}%)")
    print(f"  Test       : {len(y_test):,}  ({int(y_test.sum())} injuries = {y_test.mean()*100:.2f}%)")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "input_dim": len(feature_cols),
        "imputer": imputer,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "cutoff_date": str(cutoff.date()),
        "y_train": y_train,
        "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary Focal Loss for severe class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    alpha : weight applied to the positive (injury) class.
            Set <= 0 to use the same auto pos_weight from cfg.
    gamma : focusing exponent - higher values down-weight easy negatives more.
            gamma=2 is the standard value from the RetinaNet paper.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        # p_t is the probability of the correct class
        p_t = torch.where(targets == 1, probs, 1.0 - probs)
        # alpha_t: alpha for positives, (1-alpha) for negatives
        alpha_t = torch.where(targets == 1,
                              torch.full_like(p_t, self.alpha),
                              torch.full_like(p_t, 1.0 - self.alpha))
        return (alpha_t * (1.0 - p_t) ** self.gamma * bce).mean()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             top_k_pct: float = 0.10) -> dict[str, float]:
    model.eval()
    logits_list, targets_list = [], []

    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        logits, _ = model(x)
        logits_list.append(logits.cpu())
        targets_list.append(y.cpu())

    logits_arr = torch.cat(logits_list).numpy()
    targets_arr = torch.cat(targets_list).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_arr))

    if len(np.unique(targets_arr)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"),
                "brier": float("nan"), "top_k_recall": float("nan")}

    roc_auc = float(roc_auc_score(targets_arr, probs))
    pr_auc = float(average_precision_score(targets_arr, probs))
    brier = float(brier_score_loss(targets_arr, probs))

    k = max(1, int(len(probs) * top_k_pct))
    top_k_idx = np.argsort(probs)[-k:]
    preds_k = np.zeros(len(probs), dtype=int)
    preds_k[top_k_idx] = 1
    true_pos = float((preds_k * targets_arr).sum())
    top_k_recall = true_pos / float(targets_arr.sum()) if targets_arr.sum() > 0 else 0.0

    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "brier": brier,
            "top_k_recall": top_k_recall}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device,
                    max_grad_norm: float) -> dict[str, float]:
    model.train()
    total_loss, total_n = 0.0, 0
    t0 = time.perf_counter()

    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_n += x.size(0)

    return {"avg_loss": total_loss / max(total_n, 1),
            "duration_s": time.perf_counter() - t0}


def train(cfg: argparse.Namespace) -> dict[str, Any]:
    # Seed everything
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_Transformer"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(run_dir / "run.log", mode="w", encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("  ATHLETE INJURY PREDICTION - TRANSFORMER TRAINER")
    logger.info("=" * 60)
    logger.info(f"  Device  : {device}")
    logger.info(f"  Epochs  : {cfg.epochs}  (patience={cfg.patience})")
    logger.info(f"  LR      : {cfg.lr}  (min={cfg.lr_min})")
    logger.info(f"  Run dir : {run_dir}")

    # Data
    logger.info("\n  Loading data ...")
    data = load_data(cfg.split_date, cfg.seq_len, cfg.batch_size, cfg.seed)
    train_loader = data["train_loader"]
    val_loader = data["val_loader"]
    input_dim = data["input_dim"]

    # Model
    model = TransformerEncoderModel(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\n  Model params : {n_params:,}")

    # Loss function
    y_train = data["y_train"]
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    if cfg.pos_weight > 0:
        pw_val = cfg.pos_weight
    else:
        pw_val = n_neg / max(n_pos, 1)

    if cfg.loss == "focal":
        criterion = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
        logger.info(
            f"  Loss         : FocalLoss(alpha={cfg.focal_alpha}, gamma={cfg.focal_gamma})"
        )
    else:
        pos_weight = torch.tensor([pw_val], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info(f"  Loss         : BCEWithLogitsLoss(pos_weight={pw_val:.1f}x)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # LR schedule: Linear Warmup -> Cosine Annealing Decay
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < cfg.warmup_epochs:
            return float(current_epoch + 1) / float(max(1, cfg.warmup_epochs))
        progress = float(current_epoch - cfg.warmup_epochs) / float(max(1, cfg.epochs - cfg.warmup_epochs))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = cfg.lr_min / max(cfg.lr, 1e-8)
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Training loop
    best_roc_auc = -1.0
    best_epoch = -1
    no_improve = 0
    log_rows: list[dict] = []
    best_model_path = run_dir / "best_model.pt"

    logger.info("\n" + "  " + "-" * 70)
    logger.info(
        f"  {'Epoch':>5} | {'Loss':>8} | {'ROC-AUC':>8} | "
        f"{'PR-AUC':>7} | {'Brier':>8} | {'Top10R':>7} | {'LR':>8} | {'s':>4}"
    )
    logger.info("  " + "-" * 70)

    epoch = 0
    for epoch in range(1, cfg.epochs + 1):
        stats = train_one_epoch(model, train_loader, optimizer, criterion,
                                device, cfg.max_grad_norm)
        scheduler.step()

        metrics = evaluate(model, val_loader, device, cfg.top_k_pct)
        lr_now = optimizer.param_groups[0]["lr"]

        improved = metrics["roc_auc"] > best_roc_auc
        flag = " <-- best" if improved else ""

        if improved:
            best_roc_auc = metrics["roc_auc"]
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "cfg": vars(cfg),
            }, best_model_path)
        else:
            no_improve += 1

        logger.info(
            f"  {epoch:>5} | {stats['avg_loss']:>8.5f} | {metrics['roc_auc']:>8.4f} | "
            f"{metrics['pr_auc']:>7.4f} | {metrics['brier']:>8.5f} | "
            f"{metrics['top_k_recall']:>7.3f} | {lr_now:>8.2e} | "
            f"{stats['duration_s']:>3.1f}s{flag}"
        )

        log_rows.append({
            "epoch": epoch, "train_loss": stats["avg_loss"],
            "val_roc_auc": metrics["roc_auc"], "val_pr_auc": metrics["pr_auc"],
            "val_brier": metrics["brier"], "val_top10_recall": metrics["top_k_recall"],
            "lr": lr_now, "epoch_time_s": stats["duration_s"], "is_best": improved,
        })

        if no_improve >= cfg.patience:
            logger.info(f"\n  Early stopping at epoch {epoch} (patience={cfg.patience}).")
            break

    # Save log
    log_path = run_dir / "train_log.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    # Load best checkpoint and do final eval
    ckpt = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    final = evaluate(model, val_loader, device, cfg.top_k_pct)

    logger.info("\n" + "=" * 60)
    logger.info("  FINAL RESULTS (best checkpoint)")
    logger.info("=" * 60)
    logger.info(f"  Best epoch  : {best_epoch} / {epoch}")
    logger.info(f"  ROC-AUC     : {final['roc_auc']:.4f}")
    logger.info(f"  PR-AUC      : {final['pr_auc']:.4f}")
    logger.info(f"  Brier       : {final['brier']:.5f}")
    logger.info(f"  Top-10% Recall : {final['top_k_recall']*100:.1f}%")
    logger.info("=" * 60)

    results: dict[str, Any] = {
        "run_id": run_id,
        "model": "Transformer",
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "n_params": n_params,
        "metrics": final,
        "config": vars(cfg),
        "run_dir": str(run_dir),
        "best_model_path": str(best_model_path),
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n  Saved -> {run_dir}\n")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Athlete Injury Prediction - Transformer Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split-date", default="2021-01-01",
                        help="Train/test cutoff date")
    parser.add_argument("--seq-len", type=int, default=14,
                        help="Lookback window in days")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Transformer hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.25,
                        help="Dropout probability")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Peak learning rate (AdamW)")
    parser.add_argument("--lr-min", type=float, default=1e-6,
                        help="Min LR for cosine annealing")
    parser.add_argument("--warmup-epochs", type=int, default=2,
                        help="Linear warmup epochs before cosine decay")
    parser.add_argument("--loss", choices=["bce", "focal"], default="focal",
                        help="Loss function: bce (BCEWithLogitsLoss) or focal (FocalLoss)")
    parser.add_argument("--focal-alpha", type=float, default=0.25,
                        help="Focal loss alpha (positive class weight, 0-1)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma focusing exponent")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW weight decay")
    parser.add_argument("--pos-weight", type=float, default=5.0,
                        help="Positive class weight in BCE (<=0 uses auto n_neg/n_pos)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--top-k-pct", type=float, default=0.10,
                        help="Top-K%% risk threshold for recall")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
