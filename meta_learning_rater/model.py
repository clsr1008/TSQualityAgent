"""
ScoreModel and BradleyTerryLoss for the meta-learning rater.

ScoreModel: MLP that maps a time-series embedding to a scalar quality score.
BradleyTerryLoss: preference learning loss based on the Bradley-Terry model.

  P(B > A) = sigmoid(score_B - score_A)
  Loss = -[ p * log P(B>A) + (1-p) * log P(A>B) ]
  where p = comparisons_avg ∈ [0, 1]
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BradleyTerryLoss(nn.Module):
    def forward(
        self,
        scores_a: torch.Tensor,
        scores_b: torch.Tensor,
        p_b_greater_a: torch.Tensor,
    ) -> torch.Tensor:
        diff = scores_b - scores_a
        log_prob = (
            torch.log(torch.sigmoid(diff)) * p_b_greater_a
            + torch.log(torch.sigmoid(-diff)) * (1.0 - p_b_greater_a)
        )
        return -log_prob.mean()


class ScoreModel(nn.Module):
    """
    MLP quality scorer.

    input_dim  : MOMENT embedding dimension (typically 1024 for MOMENT-1-base)
    hidden_dim : width of hidden layers
    num_layers : total number of linear layers (including input and output)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        assert num_layers >= 2, "num_layers must be at least 2"

        self.input_layer  = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
        )
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers - 2)]
        )
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.activation   = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.input_layer(x))
        for layer, norm in zip(self.hidden_layers, self.norm_layers):
            residual = x
            x = self.activation(norm(layer(x)))
            x = x + residual
        return self.output_layer(x).squeeze(-1)