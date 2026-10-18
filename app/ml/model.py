"""
Deep Recurrent Neural Network (LSTM / GRU) with Temporal Attention
=================================================================
Implements Flowchart Steps:
- Input: Previous Days of Athlete Data [Batch, Seq_Len, Num_Features]
- LSTM or GRU Neural Network
- Temporal Attention Layer for Explainability (Day-level attribution)
- Calibrated Probability of Future Injury Report
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Computes attention weights across each day in the historical lookback sequence.
    This directly reveals WHICH DAYS contributed most to the elevated injury risk.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, rnn_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # rnn_outputs: (batch_size, seq_len, hidden_dim)
        scores = self.projection(rnn_outputs)  # (batch_size, seq_len, 1)
        weights = F.softmax(scores, dim=1)     # (batch_size, seq_len, 1)
        context = torch.sum(weights * rnn_outputs, dim=1)  # (batch_size, hidden_dim)
        return context, weights.squeeze(-1)    # (batch_size, hidden_dim), (batch_size, seq_len)


class AthleteInjuryRNN(nn.Module):
    """Bidirectional / Multilayer Recurrent Network with Temporal Attention.
    Supports either GRU or LSTM backbones.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        rnn_type: str = "GRU",
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.upper()

        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}. Choose 'GRU' or 'LSTM'.")

        self.attention = TemporalAttention(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            logits: Tensor of shape (batch_size,) - unnormalized log-odds of injury
            attn_weights: Tensor of shape (batch_size, seq_len) - day importance weights
        """
        # rnn_out: (batch_size, seq_len, hidden_dim)
        rnn_out, _ = self.rnn(x)
        context, attn_weights = self.attention(rnn_out)
        logits = self.classifier(context).squeeze(-1)
        return logits, attn_weights


class TemperatureCalibrator(nn.Module):
    """Post-hoc probability calibration via Temperature Scaling.
    Transforms raw network logits z into calibrated probabilities p = sigmoid(z / T).
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.clamp(self.temperature, min=0.01)
        return torch.sigmoid(logits / temp)

    def calibrate(self, val_logits: torch.Tensor, val_targets: torch.Tensor, max_iter: int = 100) -> float:
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.05, max_iter=max_iter)
        bce = nn.BCEWithLogitsLoss()

        def eval_loss():
            optimizer.zero_grad()
            temp = torch.clamp(self.temperature, min=0.01)
            loss = bce(val_logits / temp, val_targets)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(torch.clamp(self.temperature, min=0.01).item())
