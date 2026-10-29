"""
Neural Network Architecture Zoo for Athlete Injury Prediction
=============================================================
Implements all 10 sequence architectures for benchmarking:
  1.  GRU                    - Gated Recurrent Unit
  2.  LSTM                   - Long Short-Term Memory
  3.  CNN1D                  - 1-D Convolutional Neural Network
  4.  TCN                    - Temporal Convolutional Network (dilated causal convs)
  5.  CNN_GRU                - CNN feature extractor -> GRU temporal integrator
  6.  CNN_LSTM               - CNN feature extractor -> LSTM temporal integrator
  7.  AttentionLSTM          - LSTM + Multi-Head Self-Attention
  8.  InceptionTime          - Inception-style multi-scale 1-D convolutions
  9.  TransformerEncoder     - Positional-encoded Transformer encoder
  10. TFT                    - Temporal Fusion Transformer (simplified)

Every model exposes the SAME interface:
    logits, attn_weights = model(x)
    x:            Tensor (batch, seq_len, input_dim)
    logits:       Tensor (batch,)           - raw log-odds (no sigmoid)
    attn_weights: Tensor (batch, seq_len)   - day-level importance
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared Helpers
# ---------------------------------------------------------------------------

class _TemporalAttention(nn.Module):
    """Soft attention that produces seq_len importance weights."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: torch.Tensor):
        # h: (B, T, H)
        scores = self.proj(h)                # (B, T, 1)
        weights = F.softmax(scores, dim=1)   # (B, T, 1)
        context = (weights * h).sum(dim=1)   # (B, H)
        return context, weights.squeeze(-1)  # (B, H), (B, T)


class _Classifier(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 1. GRU
# ---------------------------------------------------------------------------

class GRUModel(nn.Module):
    """GRU + Temporal Attention."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = _TemporalAttention(hidden_dim)
        self.classifier = _Classifier(hidden_dim, dropout)

    def forward(self, x):
        h, _ = self.gru(x)
        context, attn = self.attention(h)
        return self.classifier(context), attn


# ---------------------------------------------------------------------------
# 2. LSTM
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    """LSTM + Temporal Attention."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = _TemporalAttention(hidden_dim)
        self.classifier = _Classifier(hidden_dim, dropout)

    def forward(self, x):
        h, _ = self.lstm(x)
        context, attn = self.attention(h)
        return self.classifier(context), attn


# ---------------------------------------------------------------------------
# 3. 1D CNN
# ---------------------------------------------------------------------------

class CNN1DModel(nn.Module):
    """Three-layer 1-D Convolutional Network with Global Average Pooling."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.classifier = _Classifier(32, dropout)

    def forward(self, x):
        xp = x.permute(0, 2, 1)             # (B, F, T)
        out = self.conv_block(xp)            # (B, 32, T)
        B, C, T = out.shape
        attn = torch.ones(B, T, device=x.device) / T
        context = out.mean(dim=-1)           # Global Average Pool
        return self.classifier(context), attn


# ---------------------------------------------------------------------------
# 4. TCN - Temporal Convolutional Network
# ---------------------------------------------------------------------------

class _TCNBlock(nn.Module):
    """Single dilated causal residual block for TCN."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.pad = pad

    def forward(self, x):
        res = x
        h = self.conv1(x)
        h = h[:, :, : -self.pad] if self.pad > 0 else h
        h = F.relu(self.bn1(h))
        h = self.drop(h)
        h = self.conv2(h)
        h = h[:, :, : -self.pad] if self.pad > 0 else h
        h = F.relu(self.bn2(h))
        h = self.drop(h)
        if self.downsample is not None:
            res = self.downsample(res)
        return F.relu(h + res)


class TCNModel(nn.Module):
    """Temporal Convolutional Network with 4 dilated causal blocks (dilations: 1,2,4,8)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        channels = [input_dim] + [hidden_dim] * 4
        dilations = [1, 2, 4, 8]
        blocks = []
        for i, d in enumerate(dilations):
            blocks.append(_TCNBlock(channels[i], channels[i + 1], kernel_size=3, dilation=d, dropout=dropout))
        self.net = nn.Sequential(*blocks)
        self.classifier = _Classifier(hidden_dim, dropout)

    def forward(self, x):
        xp = x.permute(0, 2, 1)
        out = self.net(xp)                   # (B, H, T)
        B, H, T = out.shape
        attn = torch.ones(B, T, device=x.device) / T
        context = out[:, :, -1]              # causal: use last timestep
        return self.classifier(context), attn


# ---------------------------------------------------------------------------
# 5. CNN-GRU Hybrid
# ---------------------------------------------------------------------------

class CNNGRUModel(nn.Module):
    """CNN extracts local patterns -> GRU models temporal dependencies."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.attention = _TemporalAttention(hidden_dim)
        self.classifier = _Classifier(hidden_dim, dropout)

    def forward(self, x):
        xp = x.permute(0, 2, 1)
        cnn_out = self.cnn(xp).permute(0, 2, 1)
        h, _ = self.gru(cnn_out)
        context, attn = self.attention(h)
        return self.classifier(context), attn


# ---------------------------------------------------------------------------
# 6. CNN-LSTM Hybrid
# ---------------------------------------------------------------------------

class CNNLSTMModel(nn.Module):
    """CNN extracts local patterns -> LSTM models temporal dependencies."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.attention = _TemporalAttention(hidden_dim)
        self.classifier = _Classifier(hidden_dim, dropout)

    def forward(self, x):
        xp = x.permute(0, 2, 1)
        cnn_out = self.cnn(xp).permute(0, 2, 1)
        h, _ = self.lstm(cnn_out)
        context, attn = self.attention(h)
        return self.classifier(context), attn


# ---------------------------------------------------------------------------
# 7. Attention-based LSTM
# ---------------------------------------------------------------------------

class AttentionLSTMModel(nn.Module):
    """LSTM + Multi-Head Self-Attention across timesteps."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.25):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                         dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool_attn = _TemporalAttention(hidden_dim)
        self.classifier = _Classifier(hidden_dim, dropout)

    def forward(self, x):
        h, _ = self.lstm(x)
        mha_out, _ = self.mha(h, h, h)
        h = self.norm(h + mha_out)
        context, attn = self.pool_attn(h)
        return self.classifier(context), attn


# ---------------------------------------------------------------------------
# 8. InceptionTime
# ---------------------------------------------------------------------------

class _InceptionBlock(nn.Module):
    """Multi-scale inception module: conv kernels [1,3,5,7] + MaxPool branch."""

    def __init__(self, in_ch: int, out_ch: int = 16, dropout: float = 0.1):
        super().__init__()
        kernels = (1, 3, 5, 7)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            )
            for k in kernels
        ])
        self.maxpool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )
        self.total_out = out_ch * (len(kernels) + 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        outs = [b(x) for b in self.branches] + [self.maxpool_branch(x)]
        return self.drop(torch.cat(outs, dim=1))


class InceptionTimeModel(nn.Module):
    """InceptionTime: 3 stacked Inception blocks + Global Average Pool."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        # Each inception block has 5 branches; derive out_ch so total ~ hidden_dim
        out_ch = max(8, hidden_dim // 5)
        b_out = out_ch * 5   # 5 branches each
        self.block1 = _InceptionBlock(input_dim, out_ch=out_ch, dropout=dropout)
        self.block2 = _InceptionBlock(b_out, out_ch=out_ch, dropout=dropout)
        self.block3 = _InceptionBlock(b_out, out_ch=out_ch, dropout=dropout)
        self.classifier = _Classifier(b_out, dropout)

    def forward(self, x):
        xp = x.permute(0, 2, 1)
        h = self.block3(self.block2(self.block1(xp)))  # (B, C, T)
        B, C, T = h.shape
        attn = torch.ones(B, T, device=x.device) / T
        context = h.mean(dim=-1)
        return self.classifier(context), attn


# ---------------------------------------------------------------------------
# 9. Time-Series Transformer Encoder
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class TransformerEncoderModel(nn.Module):
    """Transformer Encoder: Input Projection + Positional Encoding + N encoder layers."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 2, dim_ff: int = 128, dropout: float = 0.25):
        super().__init__()
        d_model = hidden_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_attn = _TemporalAttention(d_model)
        self.classifier = _Classifier(d_model, dropout)

    def forward(self, x):
        h = self.pos_enc(self.input_proj(x))
        h = self.encoder(h)
        context, attn = self.pool_attn(h)
        return self.classifier(context), attn


# ---------------------------------------------------------------------------
# 10. Temporal Fusion Transformer (TFT) - Simplified
# ---------------------------------------------------------------------------

class _GatedResidual(nn.Module):
    """Gated Residual Network (GRN) as used in the TFT paper."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.elu(self.linear1(x))
        h = self.drop(self.linear2(h))
        g = torch.sigmoid(self.gate(x))
        return self.norm(x + g * h)


class TFTModel(nn.Module):
    """
    Simplified Temporal Fusion Transformer.
    Embedding -> GRN variable selection -> LSTM encoder
    -> Multi-Head Self-Attention -> GRN skip -> mean pool -> classifier
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.25):
        super().__init__()
        d_model = hidden_dim
        self.embedding = nn.Linear(input_dim, d_model)
        self.grn_select = _GatedResidual(d_model, dropout)
        self.lstm_enc = nn.LSTM(d_model, d_model, num_layers=2, batch_first=True, dropout=dropout)
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.grn_post = _GatedResidual(d_model, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.pool_attn = _TemporalAttention(d_model)
        self.classifier = _Classifier(d_model, dropout)

    def forward(self, x):
        h = self.grn_select(self.embedding(x))
        h, _ = self.lstm_enc(h)
        attn_out, _ = self.mha(h, h, h)
        h = self.grn_post(self.norm(h + attn_out))
        context, attn = self.pool_attn(h)
        return self.classifier(context), attn


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "GRU":             GRUModel,
    "LSTM":            LSTMModel,
    "CNN1D":           CNN1DModel,
    "TCN":             TCNModel,
    "CNN-GRU":         CNNGRUModel,
    "CNN-LSTM":        CNNLSTMModel,
    "Attention-LSTM":  AttentionLSTMModel,
    "InceptionTime":   InceptionTimeModel,
    "Transformer":     TransformerEncoderModel,
    "TFT":             TFTModel,
}
