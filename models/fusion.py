"""Feature-level fusion for video, holistic consistency, and optional audio."""

import torch
from torch import nn


class EvidenceFusion(nn.Module):
    """Fuse modality vectors while excluding absent modalities from the gate.

    Inputs may be pooled vectors ``(B,D)`` or temporal sequences ``(B,T,D)``.
    Audio is optional until its checkpoint and preprocessing are available.
    """

    def __init__(self, video_dim: int, hvcn_dim: int, audio_dim: int | None, fusion_dim: int = 512):
        """Set feature sizes; omit the audio projection when audio is disabled."""
        super().__init__()
        self.video_projection = nn.Linear(video_dim, fusion_dim)
        self.hvcn_projection = nn.Linear(hvcn_dim, fusion_dim)
        self.audio_projection = nn.Linear(audio_dim, fusion_dim) if audio_dim is not None else None
        self.gate = nn.Linear(fusion_dim, 1)
        self.output = nn.Sequential(nn.LayerNorm(fusion_dim), nn.Linear(fusion_dim, fusion_dim), nn.GELU())

    def forward(
        self,
        video: torch.Tensor,
        hvcn: torch.Tensor,
        audio: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project modalities and return their learned, presence-aware mixture."""
        if video.shape[:-1] != hvcn.shape[:-1]:
            raise ValueError("Video and HVCN inputs must have matching batch/time dimensions")
        features = [self.video_projection(video), self.hvcn_projection(hvcn)]
        present = [torch.ones(video.shape[:-1], dtype=torch.bool, device=video.device)] * 2
        if audio is not None:
            if self.audio_projection is None:
                raise ValueError("audio_dim must be configured to fuse audio features")
            if audio.shape[:-1] != video.shape[:-1]:
                raise ValueError("Audio and video inputs must have matching batch/time dimensions")
            features.append(self.audio_projection(audio))
            mask = torch.ones(video.shape[:-1], dtype=torch.bool, device=video.device)
            if audio_mask is not None:
                if audio_mask.shape != mask.shape:
                    raise ValueError("audio_mask must match batch/time dimensions")
                mask = audio_mask.to(device=video.device, dtype=torch.bool)
            present.append(mask)

        stacked = torch.stack(features, dim=-2)
        present_mask = torch.stack(present, dim=-1)
        logits = self.gate(torch.tanh(stacked)).squeeze(-1)
        # Missing modalities must receive zero weight after the softmax.
        logits = logits.masked_fill(~present_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        return self.output((stacked * weights).sum(dim=-2))

