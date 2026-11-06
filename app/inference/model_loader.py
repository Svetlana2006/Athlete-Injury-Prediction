"""
ATHLETEIQ – Inference Engine
==============================
Loads the best Transformer checkpoint once at startup and exposes
a predict() function that maps a raw 2-D feature sequence to a
sigmoid injury-risk probability.

Preprocessing contract (must match training):
  - 22 RAW_FEATURE_COLUMNS in the exact order defined in app/ml/train.py
  - seq_len = 14 (lookback window)
  - SimpleImputer(strategy='median') → StandardScaler
    (parameters recovered from the checkpoint saved at training time)
  - Input tensor shape : (1, seq_len, 22)
  - Output             : sigmoid(logits) → float in [0, 1]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = (
    BASE_DIR
    / "artifacts"
    / "runs"
    / "20261104_153047_Transformer"
    / "best_model.pt"
)

# ---------------------------------------------------------------------------
# Feature contract (matches app/ml/train.py RAW_FEATURE_COLUMNS)
# ---------------------------------------------------------------------------
RAW_FEATURE_COLUMNS: list[str] = [
    "rpe", "duration_minutes", "session_rpe", "daily_load", "weekly_load",
    "fatigue", "soreness", "mood", "readiness",
    "sleep_duration", "sleep_quality", "stress",
    "total_distance_m", "high_speed_distance_m", "sprint_distance_m",
    "max_speed_m_s", "mean_speed_m_s",
    "prior_injury_count", "has_prior_injury", "is_rest_day", "has_gps",
    "illness_last_7d",
]

SEQ_LEN: int = 14
INPUT_DIM: int = len(RAW_FEATURE_COLUMNS)   # 22

# ---------------------------------------------------------------------------
# Module-level singletons (initialised lazily via load_model())
# ---------------------------------------------------------------------------
_model: "torch.nn.Module | None" = None
_imputer_medians: "np.ndarray | None" = None   # shape (22,)
_scaler_mean: "np.ndarray | None" = None        # shape (22,)
_scaler_scale: "np.ndarray | None" = None       # shape (22,)
_device: "torch.device | None" = None
_model_meta: dict[str, Any] = {}

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_model() -> None:
    """
    Load the trained Transformer checkpoint into module-level singletons.
    Safe to call multiple times – subsequent calls are no-ops.
    """
    global _model, _imputer_medians, _scaler_mean, _scaler_scale, _device, _model_meta

    if _model is not None:
        return

    # Add repo root so app.ml.nn_models can be imported
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Make sure the training run has completed successfully."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[ModelLoader] Loading checkpoint from {CHECKPOINT_PATH} on {device}")

    ckpt: dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    from app.ml.nn_models import TransformerEncoderModel

    cfg: dict = ckpt.get("cfg", {})
    hidden_dim: int = cfg.get("hidden_dim", 64)
    dropout: float = cfg.get("dropout", 0.25)

    checkpoint_features = ckpt.get("feature_cols")
    if checkpoint_features is not None and checkpoint_features != RAW_FEATURE_COLUMNS:
        raise ValueError(
            "Checkpoint feature order does not match the inference feature contract."
        )

    preprocessing_keys = ("imputer_medians", "scaler_mean", "scaler_scale")
    has_preprocessing_stats = all(key in ckpt for key in preprocessing_keys)

    # Recover preprocessing parameters saved alongside the checkpoint
    if has_preprocessing_stats:
        imputer_medians = np.asarray(ckpt["imputer_medians"], dtype=np.float64)
        scaler_mean = np.asarray(ckpt["scaler_mean"], dtype=np.float64)
        scaler_scale = np.asarray(ckpt["scaler_scale"], dtype=np.float64)
        if any(
            values.shape != (INPUT_DIM,)
            for values in (imputer_medians, scaler_mean, scaler_scale)
        ):
            raise ValueError(
                f"Checkpoint preprocessing statistics must each contain {INPUT_DIM} values."
            )
        logger.info("[ModelLoader] Recovered imputer + scaler from checkpoint.")
    else:
        logger.warning(
            "[ModelLoader] Checkpoint does not contain preprocessing stats. "
            "Falling back to per-sequence z-score. Predictions are illustrative."
        )
        imputer_medians = None
        scaler_mean = None
        scaler_scale = None

    model = TransformerEncoderModel(
        input_dim=INPUT_DIM,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _model = model
    _imputer_medians = imputer_medians
    _scaler_mean = scaler_mean
    _scaler_scale = scaler_scale
    _device = device
    _model_meta = {
        "run_id": "20261104_153047_Transformer",
        "model_version": "Transformer-Focal-v1",
        "roc_auc": 0.9191,
        "pr_auc": 0.0226,
        "seq_len": SEQ_LEN,
        "input_dim": INPUT_DIM,
        "loss": "FocalLoss(alpha=0.25, gamma=2.0)",
        "device": str(device),
        "preprocessing_mode": (
            "training_statistics" if has_preprocessing_stats else "per_sequence_zscore"
        ),
    }
    logger.info(f"[ModelLoader] Model ready. ROC-AUC={_model_meta['roc_auc']}")


def is_loaded() -> bool:
    return _model is not None


def get_meta() -> dict[str, Any]:
    return _model_meta.copy()


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _impute(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    for col in range(out.shape[1]):
        nan_mask = np.isnan(out[:, col])
        if nan_mask.any():
            fill = (
                _imputer_medians[col]
                if _imputer_medians is not None
                else float(np.nanmedian(out[:, col]) if not nan_mask.all() else 0.0)
            )
            out[nan_mask, col] = fill
    return out


def _scale(arr: np.ndarray) -> np.ndarray:
    if _scaler_mean is not None and _scaler_scale is not None:
        scale = np.where(_scaler_scale == 0, 1.0, _scaler_scale)
        return (arr - _scaler_mean) / scale
    # Fallback: per-column z-score over the provided sequence
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    std  = np.where(std == 0, 1.0, std)
    return (arr - mean) / std


def preprocess_sequence(raw_sequence: list[list[float]]) -> np.ndarray:
    """
    Convert a list-of-lists (shape [seq_len, 22]) into a scaled numpy array.
    """
    arr = np.array(raw_sequence, dtype=np.float64)
    if arr.shape != (SEQ_LEN, INPUT_DIM):
        raise ValueError(
            f"Expected sequence shape ({SEQ_LEN}, {INPUT_DIM}), got {arr.shape}."
        )
    arr = _impute(arr)
    arr = _scale(arr)
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(raw_sequence: list[list[float]]) -> dict[str, Any]:
    """
    Run one forward pass on a single athlete sequence.

    Parameters
    ----------
    raw_sequence : list[list[float]]
        Shape [SEQ_LEN, INPUT_DIM] - values in original (unscaled) units.

    Returns
    -------
    dict with keys:
        predicted_probability  float  [0, 1]
        risk_category          str    Low | Moderate | Elevated | High
        attention_weights      list   per-day attention weights (len=SEQ_LEN)
        model_version          str
        prediction_source      str
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    arr = preprocess_sequence(raw_sequence)
    x   = torch.from_numpy(arr).unsqueeze(0).to(_device)

    logits, attn = _model(x)
    prob = float(torch.sigmoid(logits).cpu().item())
    attn_weights = attn.cpu().squeeze(0).tolist()

    if prob < 0.15:
        category = "Low"
    elif prob < 0.30:
        category = "Moderate"
    elif prob < 0.50:
        category = "Elevated"
    else:
        category = "High"

    return {
        "predicted_probability": round(prob, 4),
        "risk_category": category,
        "attention_weights": attn_weights,
        "model_version": _model_meta.get("model_version", "Transformer-Focal-v1"),
        "prediction_source": "actual_model",
    }
