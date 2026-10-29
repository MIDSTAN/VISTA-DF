"""Add position-aware self-attention to ordered per-frame feature sequences."""

import torch
from torch import nn


class TemporalTransformer(nn.Module):
    """Model relationships between frames while optionally hiding future frames."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_layers: int = 2,
        num_heads: int = 8,
        max_frames: int = 32,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        """Create attention layers for sequences up to ``max_frames``."""
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.position = nn.Parameter(torch.zeros(1, max_frames, hidden_size))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.causal = causal

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Transform ``(B,T,D)`` features; ``True`` mask entries are padding."""
        if x.ndim != 3:
            raise ValueError(f"Expected (B,T,D), got {tuple(x.shape)}")
        if x.shape[1] > self.position.shape[1]:
            raise ValueError(f"Sequence length {x.shape[1]} exceeds max_frames={self.position.shape[1]}")
        if padding_mask is not None and tuple(padding_mask.shape) != tuple(x.shape[:2]):
            raise ValueError("padding_mask must have shape (B,T)")
        x = x + self.position[:, : x.shape[1]].to(dtype=x.dtype, device=x.device)
        attention_mask = None
        if self.causal:
            # Prevent the next-frame prediction objective from seeing its target.
            attention_mask = torch.triu(
                torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), diagonal=1
            )
        return self.norm(
            self.encoder(x, mask=attention_mask, src_key_padding_mask=padding_mask)
        )

