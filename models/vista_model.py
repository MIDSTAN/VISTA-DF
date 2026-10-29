"""Video-only temporal representation model for Kinetics pretraining."""

import torch
from torch import nn

from models.fusion import EvidenceFusion
from models.hvcn import HolisticVisualConsistencyNetwork
from models.temporal_transformer import TemporalTransformer
from models.video_network import VideoFeatureExtractor


class VistaTemporalModel(nn.Module):
    """Frozen VideoMAE plus trainable temporal consistency heads.

    The model returns a prediction for each next-frame representation. This is
    a representation-learning model, not a real/fake classifier.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        encoder_trainable: bool = False,
        audio_dim: int | None = None,
    ):
        super().__init__()
        self.video = VideoFeatureExtractor(checkpoint_dir, trainable=encoder_trainable)
        dim = self.video.hidden_size
        self.hvcn = HolisticVisualConsistencyNetwork(dim)
        self.temporal = TemporalTransformer(dim, num_layers=2, num_heads=8, max_frames=32)
        self.fusion = EvidenceFusion(dim, dim, audio_dim=audio_dim, fusion_dim=dim)
        self.predictor = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(
        self,
        pixel_values: torch.Tensor,
        audio_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return per-frame features and next-frame predictions.

        Pixels have shape ``(B,T,C,H,W)``. Visual, HVCN, temporal, and fused
        features retain ``(B,T,D)``; prediction and target are ``(B,T-1,D)``.
        """
        batch, frames = pixel_values.shape[:2]
        patch_tokens = self.video(pixel_values)
        if patch_tokens.shape[1] % frames:
            raise RuntimeError(
                f"VideoMAE returned {patch_tokens.shape[1]} tokens, not divisible by {frames} frames"
            )
        # Spatially pool patch tokens independently per frame, preserving time.
        visual = patch_tokens.reshape(batch, frames, -1, patch_tokens.shape[-1]).mean(dim=2)
        hvcn = self.hvcn(visual)
        temporal = self.temporal(visual)
        if audio_features is not None:
            if audio_features.ndim == 3:
                audio_features = audio_features.mean(dim=1)
            if audio_features.ndim != 2 or audio_features.shape[0] != batch:
                raise ValueError("audio_features must have shape (B,D) or (B,L,D)")
            audio_features = audio_features.unsqueeze(1).expand(-1, frames, -1)
        fused = self.fusion(temporal, hvcn, audio=audio_features)
        # Each context frame predicts the next visual feature; the final frame
        # has no later target in this clip.
        prediction = self.predictor(fused[:, :-1])
        return {
            "visual_features": visual,
            "hvcn_features": hvcn,
            "temporal_features": temporal,
            "fused_features": fused,
            "prediction": prediction,
            "target": visual[:, 1:].detach(),
        }

