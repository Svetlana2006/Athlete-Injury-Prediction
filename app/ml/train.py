"""
Deep Sequential Model Training & Evaluation Pipeline
====================================================
Implements Flowchart:
1. Build historical sequences for each athlete (sliding lookback window L = 14 days)
2. Train Recurrent Neural Network (GRU or LSTM) with Temporal Attention
3. BCEWithLogitsLoss with pos_weight for severe class imbalance (~1.4%)
4. Evaluate on Unseen Future Holdout Set (PR-AUC, ROC-AUC, Brier score, Top-10% Recall)
5. Post-hoc Temperature Scaling Calibration
6. Extract Temporal Attention & Feature Attribution for Dashboard
7. Save production bundle to artifacts/deep_injury_model.pt

Usage:
    python app/ml/train.py
    python app/ml/train.py --rnn-type LSTM --epochs 15 --batch-size 128
"""

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from app.ml.dataset import prepare_sequential_data
from app.ml.model import AthleteInjuryRNN, TemperatureCalibrator

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "daily_observations.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Runs inference and extracts logits, targets, attention weights, athlete IDs, and dates."""
    model.eval()
    all_logits = []
    all_targets = []
    all_attns = []
    all_athletes = []
    all_dates = []

    with torch.no_grad():
        for seqs, targets, athletes, dates in loader:
            seqs = seqs.to(device)
            logits, attn = model(seqs)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.numpy())
            all_attns.append(attn.cpu().numpy())
            all_athletes.extend(athletes)
            all_dates.extend(dates)

    logits_arr = np.concatenate(all_logits)
    targets_arr = np.concatenate(all_targets)
    attns_arr = np.concatenate(all_attns)

    return logits_arr, targets_arr, attns_arr, all_athletes, all_dates


def compute_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Computes PR-AUC, ROC-AUC, Brier Score, and Top-10% Triage Recall."""
    pr_auc = float(average_precision_score(y_true, probs))
    roc_auc = float(roc_auc_score(y_true, probs))
    brier = float(brier_score_loss(y_true, probs))

    # Top 10% triage threshold
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
        "baseline_injury_rate": float(y_true.mean()),
    }


def train_deep_pipeline(
    rnn_type: str = "GRU",
    seq_len: int = 14,
    hidden_dim: int = 64,
    num_layers: int = 2,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    split_ratio: float = 0.80,
    split_date: str | None = None,
    data_path: Path | None = None,
    artifacts_dir: Path | None = None,
) -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    art_dir = artifacts_dir or ARTIFACTS_DIR

    # 1. Build Sequences and Prepare Time-Aware Datasets
    data_dict = prepare_sequential_data(
        filepath=data_path or DATA_PATH,
        seq_len=seq_len,
        split_ratio=split_ratio,
        split_date=split_date,
    )

    train_ds = data_dict["train_dataset"]
    test_ds = data_dict["test_dataset"]
    feature_cols = data_dict["feature_cols"]

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 2. Imbalance-Aware Loss Function (pos_weight)
    y_train_np = train_ds.targets.numpy()
    num_neg = (y_train_np == 0).sum()
    num_pos = (y_train_np == 1).sum()
    pos_weight_val = max(1.0, float(num_neg / max(1, num_pos)))
    print(f"Calculated positive class loss weight: {pos_weight_val:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))

    # 3. Model & Optimizer
    model = AthleteInjuryRNN(
        input_dim=len(feature_cols),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        rnn_type=rnn_type,
        dropout=0.25,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\n" + "=" * 62)
    print(f"TRAINING {rnn_type.upper()} + TEMPORAL ATTENTION ({epochs} EPOCHS)")
    print("=" * 62)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for seqs, targets, _, _ in train_loader:
            seqs = seqs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits, _ = model(seqs)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_loss += loss.item() * len(targets)

        scheduler.step()
        avg_train_loss = total_loss / len(train_ds)

        # Quick validation evaluation
        if epoch % 5 == 0 or epoch == epochs:
            test_logits, test_y, _, _, _ = evaluate_model(model, test_loader, device)
            test_probs = 1.0 / (1.0 + np.exp(-test_logits))
            metrics = compute_metrics(test_y, test_probs)
            print(
                f"Epoch [{epoch:2d}/{epochs:2d}] "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Holdout PR-AUC: {metrics['pr_auc']:.4f} | "
                f"ROC-AUC: {metrics['roc_auc']:.4f} | "
                f"Top-10% Recall: {metrics['top10_recall']*100:.1f}%"
            )

    # 4. Calibration on Holdout Logits
    print("\n" + "=" * 62)
    print("POST-HOC TEMPERATURE SCALING PROBABILITY CALIBRATION")
    print("=" * 62)

    test_logits, test_y, test_attns, test_athletes, test_dates = evaluate_model(model, test_loader, device)
    raw_probs = 1.0 / (1.0 + np.exp(-test_logits))

    calibrator = TemperatureCalibrator()
    learned_temp = calibrator.calibrate(
        torch.tensor(test_logits, dtype=torch.float32),
        torch.tensor(test_y, dtype=torch.float32),
    )

    with torch.no_grad():
        calibrated_probs = calibrator(torch.tensor(test_logits, dtype=torch.float32)).numpy()

    raw_metrics = compute_metrics(test_y, raw_probs)
    cal_metrics = compute_metrics(test_y, calibrated_probs)

    print(f"Learned Temperature parameter (T) : {learned_temp:.4f}")
    print(f"Raw Brier Score                   : {raw_metrics['brier_score']:.4f}")
    print(f"Calibrated Brier Score            : {cal_metrics['brier_score']:.4f} (lower is better)")
    print(f"Calibrated PR-AUC                 : {cal_metrics['pr_auc']:.4f}")
    print(f"Calibrated ROC-AUC                : {cal_metrics['roc_auc']:.4f}")
    print(f"Calibrated Probabilities Range   : {calibrated_probs.min()*100:.2f}% to {calibrated_probs.max()*100:.2f}%")
    print(f"Injuries Caught in Top 10% Alerts : {cal_metrics['top10_recall']*100:.1f}%")

    # 5. Extract Average Attention Weight Distribution across Lookback Window
    avg_attention = test_attns.mean(axis=0)
    print("\n" + "=" * 62)
    print(f"TEMPORAL ATTENTION WEIGHTS (LOOKBACK WINDOW: {seq_len} DAYS)")
    print("=" * 62)
    print("Day offset (0 = Prediction Date, -13 = 13 days ago):")
    for d_idx, w in enumerate(avg_attention):
        offset = d_idx - (seq_len - 1)
        bar = "=" * int(w * 100)
        print(f"   Day {offset:+3d} : {w*100:5.2f}% {bar}")

    # 6. Save Model Bundle
    art_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = art_dir / "deep_injury_model.pt"
    results_json_path = art_dir / "deep_model_results.json"

    export_bundle = {
        "model_state_dict": model.state_dict(),
        "calibrator_state_dict": calibrator.state_dict(),
        "learned_temperature": learned_temp,
        "rnn_type": rnn_type,
        "input_dim": len(feature_cols),
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "seq_len": seq_len,
        "feature_cols": feature_cols,
        "imputer": data_dict["imputer"],
        "scaler": data_dict["scaler"],
        "cutoff_date": data_dict["cutoff_date"],
        "raw_metrics": raw_metrics,
        "calibrated_metrics": cal_metrics,
        "average_attention": avg_attention.tolist(),
    }

    torch.save(export_bundle, model_save_path)
    print(f"\nSaved production PyTorch model bundle to {model_save_path}")

    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "architecture": f"{rnn_type.upper()} + Temporal Attention",
                "lookback_sequence_length": seq_len,
                "cutoff_date": data_dict["cutoff_date"],
                "learned_temperature": learned_temp,
                "metrics": cal_metrics,
                "attention_distribution": {
                    f"day_{i - seq_len + 1}": float(round(w, 4))
                    for i, w in enumerate(avg_attention)
                },
            },
            f,
            indent=2,
        )
    print(f"Saved model evaluation summary to {results_json_path}")
    print("\n" + "=" * 62)
    print("DEEP LEARNING PIPELINE COMPLETE!")
    print("=" * 62)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train deep sequential athlete injury model.")
    parser.add_argument("--rnn-type", type=str, default="GRU", choices=["GRU", "LSTM"], help="Recurrent backbone type")
    parser.add_argument("--seq-len", type=int, default=14, help="Sliding lookback sequence length in days")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden state dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of recurrent layers")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--split-ratio", type=float, default=0.80, help="Train/test temporal split ratio")
    parser.add_argument("--split-date", type=str, default=None, help="Explicit cutoff date YYYY-MM-DD")
    parser.add_argument("--data-path", type=str, default=None, help="Path to daily_observations.csv")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Path to artifacts output directory")
    args = parser.parse_args()

    train_deep_pipeline(
        rnn_type=args.rnn_type,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        split_ratio=args.split_ratio,
        split_date=args.split_date,
        data_path=Path(args.data_path) if args.data_path else None,
        artifacts_dir=Path(args.artifacts_dir) if args.artifacts_dir else None,
    )
