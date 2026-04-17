import pandas as pd
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
            weight = self.class_weights[i].to(logits.device) if self.class_weights is not None else None
            loss = F.cross_entropy(
                logits[:, i, :],
                targets[:, i],
                weight=weight,
                label_smoothing=self.label_smoothing,
            )
            total_loss = total_loss + loss
        return total_loss / self.num_emotions


def _compute_class_weights(train_csv: str, emotion_columns: List[str], num_classes: int) -> List[List[float]]:
    """
    Compute per-class weights for each emotion from the training CSV.

    For each emotion and each class c:
        weight[c] = total_samples / (num_classes * count[c])

    This balances the contribution of each class regardless of frequency.
    """
    df = pd.read_csv(train_csv)
    n = len(df)
    weights = []
    for col in emotion_columns:
        col_weights = []
        for c in range(num_classes):
            count = (df[col] == c).sum()
            w = n / (num_classes * count) if count > 0 else 1.0
            col_weights.append(float(w))
        weights.append(col_weights)
        print(f"  {col}: weights={[f'{w:.3f}' for w in col_weights]}")
    return weights


EMOTION_COLUMNS = ["Boredom", "Engagement", "Confusion", "Frustration"]


def create_loss_function(config: dict) -> nn.Module:
    loss_config = config.get('loss', {})
    num_emotions = config['model']['num_emotions']
    num_classes = config['model'].get('num_classes', 4)
    label_smoothing = loss_config.get('label_smoothing', 0.0)

    raw_weights = loss_config.get('class_weights', None)

    if raw_weights == 'auto':
        train_csv = config['data']['train_csv']
        print("Computing class weights from training data...")
        raw_weights = _compute_class_weights(train_csv, EMOTION_COLUMNS, num_classes)

    return MultiHeadCrossEntropyLoss(num_emotions, raw_weights, label_smoothing)
