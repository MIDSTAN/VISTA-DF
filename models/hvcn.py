"""Holistic temporal visual consistency features.

The module operates on spatially pooled VideoMAE frame tokens, so its signal
includes scene-wide appearance changes rather than face-only crops.
"""

import torch
from torch import nn


class HolisticVisualConsistencyNetwork(nn.Module):
    """Combine each frame's scene content with its change from the prior frame."""

    def __init__(self, hidden_size: int, output_size: int | None = None, dropout: float = 0.1):
        super().__init__()
        output_size = output_size or hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.delta_projection = nn.Sequential(
            nn.Linear(hidden_size, output_size), nn.GELU(), nn.Dropout(dropout)
        )
        self.content_projection = nn.Linear(hidden_size, output_size)
        self.fuse = nn.Sequential(
            nn.Linear(output_size * 2, output_size), nn.GELU(), nn.LayerNorm(output_size)
        )

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """Map ``(B,T,D)`` scene features to ``(B,T,D_out)`` consistency features."""
        if frame_features.ndim != 3:
            raise ValueError(f"Expected (B,T,D), got {tuple(frame_features.shape)}")
        x = self.norm(frame_features)
        delta = torch.zeros_like(x)
        if x.shape[1] > 1:
            # The first frame has no predecessor, so its change starts at zero.
            delta[:, 1:] = x[:, 1:] - x[:, :-1]
        return self.fuse(torch.cat((self.content_projection(x), self.delta_projection(delta)), dim=-1))

