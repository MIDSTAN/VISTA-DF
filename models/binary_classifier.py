"""Binary classification head for frozen VISTA-DF video representations."""

import torch
from torch import nn


class BinaryClassificationHead(nn.Module):
    """Map pooled temporal features to REAL/DEEPFAKE logits."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError(f"Expected pooled features with shape (B,D), got {tuple(features.shape)}")
        return self.layers(features)

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits.float(), dim=-1)