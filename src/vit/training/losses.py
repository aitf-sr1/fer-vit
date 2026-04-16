import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class MultiHeadCrossEntropyLoss(nn.Module):
    """Cross-entropy loss averaged across all emotion heads."""

    def __init__(
        self,
        num_emotions: int = 4,
        class_weights: Optional[List[List[float]]] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.num_emotions = num_emotions
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            self.class_weights = nn.ParameterList([
                nn.Parameter(torch.tensor(w, dtype=torch.float32), requires_grad=False)
                for w in class_weights
            ])
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (batch, num_emotions, num_classes)
        # targets: (batch, num_emotions) as torch.long
        total_loss = torch.tensor(0.0, device=logits.device)
        for i in range(self.num_emotions):
            weight = self.class_weights[i] if self.class_weights is not None else None
            loss = F.cross_entropy(
                logits[:, i, :],
                targets[:, i],
                weight=weight,
                label_smoothing=self.label_smoothing,
            )
            total_loss = total_loss + loss
        return total_loss / self.num_emotions


def create_loss_function(config: dict) -> nn.Module:
    loss_config = config.get('loss', {})
    num_emotions = config['model']['num_emotions']
    class_weights = loss_config.get('class_weights', None)
    label_smoothing = loss_config.get('label_smoothing', 0.0)
    return MultiHeadCrossEntropyLoss(num_emotions, class_weights, label_smoothing)
