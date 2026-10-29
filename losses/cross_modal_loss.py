"""Optional contrastive objective for matching pooled audio and video features.

This helper is available for a later multimodal training stage and is not used
by the current temporal-prediction trainer.
"""

import torch
import torch.nn.functional as F


def paired_audio_video_info_nce(
    video_features: torch.Tensor,
    audio_features: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric in-batch InfoNCE; rows must refer to aligned video/audio clips.

    Requires at least two distinct clips per batch to provide negatives. It is
    deliberately not used in the current video-only training stage.
    """
    if video_features.ndim != 2 or audio_features.ndim != 2:
        raise ValueError("Expected pooled modality features of shape (B,D)")
    if video_features.shape[0] != audio_features.shape[0] or video_features.shape[0] < 2:
        raise ValueError("Paired InfoNCE requires matching batches with at least two clips")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    video = F.normalize(video_features.float(), dim=-1)
    audio = F.normalize(audio_features.float(), dim=-1)
    logits = video @ audio.transpose(0, 1) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

