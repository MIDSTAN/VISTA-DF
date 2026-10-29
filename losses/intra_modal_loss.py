"""Loss for predicting future visual features from earlier video frames."""

import torch
import torch.nn.functional as F


def temporal_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    cosine_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Predict frozen VideoMAE features at t+1 from the preceding context.

    MSE preserves feature magnitude; cosine distance encourages semantic
    direction agreement. The target encoder is frozen in the initial stage.
    This learns temporal representation prediction, not manipulation labels.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target mismatch: {prediction.shape} vs {target.shape}")
    mse = F.mse_loss(prediction.float(), target.float())
    cosine = (1.0 - F.cosine_similarity(prediction.float(), target.float(), dim=-1)).mean()
    total = mse + cosine_weight * cosine
    return total, {"mse": mse.detach(), "cosine_distance": cosine.detach()}

