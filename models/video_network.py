"""Hugging Face VideoMAE feature extractor."""

from pathlib import Path

import torch
from torch import nn
from transformers import VideoMAEImageProcessor, VideoMAEModel


class VideoFeatureExtractor(nn.Module):
    """Loads the encoder portion of a local VideoMAE checkpoint.

    The encoder is frozen by default for T4-friendly staged training. A local
    Kaggle checkpoint path is required; this class never downloads weights.
    """

    def __init__(self, checkpoint_dir: str | Path, *, trainable: bool = False):
        super().__init__()
        checkpoint_dir = Path(checkpoint_dir)
        if not (checkpoint_dir / "config.json").is_file():
            raise FileNotFoundError(f"VideoMAE config not found in {checkpoint_dir}")
        self.encoder = VideoMAEModel.from_pretrained(str(checkpoint_dir), local_files_only=True)
        self.hidden_size = int(self.encoder.config.hidden_size)
        self.num_frames = int(self.encoder.config.num_frames)
        self.image_size = int(self.encoder.config.image_size)
        self.image_processor = VideoMAEImageProcessor.from_pretrained(
            str(checkpoint_dir), local_files_only=True
        )
        self.set_trainable(trainable)

    def set_trainable(self, trainable: bool) -> None:
        """Enable or freeze encoder gradients for staged training."""
        self.trainable = bool(trainable)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(self.trainable)
        if not self.trainable:
            self.encoder.eval()

    def train(self, mode: bool = True):
        """Keep a frozen encoder in evaluation mode even when its parent trains."""
        super().train(mode)
        if not self.trainable:
            self.encoder.eval()
        return self

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode input ``(B, T, C, H, W)`` into patch tokens ``(B, N, D)``."""
        if pixel_values.ndim != 5:
            raise ValueError(f"Expected (B,T,C,H,W), got {tuple(pixel_values.shape)}")
        if pixel_values.shape[1] != self.num_frames:
            raise ValueError(f"Checkpoint expects {self.num_frames} frames, got {pixel_values.shape[1]}")
        context = torch.enable_grad() if self.trainable else torch.no_grad()
        with context:
            result = self.encoder(pixel_values=pixel_values, return_dict=True)
        return result.last_hidden_state

