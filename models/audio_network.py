"""Local Whisper encoder features with checkpoint-matched log-Mel processing."""

from pathlib import Path

import torch
from torch import nn
from transformers import WhisperFeatureExtractor as HFWhisperFeatureExtractor
from transformers import WhisperModel


class WhisperFeatureExtractor(nn.Module):
    """Extract Whisper encoder features from mono 16 kHz waveforms.

    No model files are downloaded. The local directory must contain a valid
    Whisper checkpoint and processor config.
    """

    def __init__(self, checkpoint_dir: str | Path, *, trainable: bool = False):
        super().__init__()
        checkpoint_dir = Path(checkpoint_dir)
        if not (checkpoint_dir / "config.json").is_file():
            raise FileNotFoundError(f"Whisper config not found in {checkpoint_dir}")
        self.processor = HFWhisperFeatureExtractor.from_pretrained(
            str(checkpoint_dir), local_files_only=True
        )
        self.model = WhisperModel.from_pretrained(str(checkpoint_dir), local_files_only=True)
        self.hidden_size = int(self.model.config.d_model)
        self.trainable = bool(trainable)
        for parameter in self.model.parameters():
            parameter.requires_grad_(self.trainable)
        if not self.trainable:
            self.model.eval()

    def train(self, mode: bool = True):
        """Keep the frozen Whisper encoder in evaluation mode during training."""
        super().train(mode)
        if not self.trainable:
            self.model.eval()
        return self

    def waveform_to_features(self, waveforms, sampling_rate: int = 16_000) -> torch.Tensor:
        """Convert a list/array of mono waveforms to Whisper log-Mel inputs."""
        if sampling_rate != 16_000:
            raise ValueError("Resample audio to 16 kHz before Whisper preprocessing")
        batch = self.processor(waveforms, sampling_rate=sampling_rate, return_tensors="pt")
        return batch.input_features

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Encode log-Mel batches shaped ``(B,80,frames)`` into ``(B,L,D)``."""
        if input_features.ndim != 3:
            raise ValueError(f"Expected Whisper log-Mel (B,80,frames), got {tuple(input_features.shape)}")
        context = torch.enable_grad() if self.trainable else torch.no_grad()
        with context:
            result = self.model.encoder(input_features=input_features, return_dict=True)
        return result.last_hidden_state

