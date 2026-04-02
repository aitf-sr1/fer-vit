import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class MultiHeadCrossEntropyLoss(nn.Module):
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
            self.class_weights = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.tensor(w, dtype=torch.float32), requires_grad=False
                    )
                    for w in class_weights
                ]
            )
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss: torch.Tensor = torch.tensor(0.0, device=logits.device)
        for i in range(self.num_emotions):
            emotion_logits = logits[:, i, :]
            emotion_targets = targets[:, i]

            if self.class_weights is not None:
                weight = self.class_weights[i]
            else:
                weight = None

            loss = F.cross_entropy(
                emotion_logits,
                emotion_targets,
                weight=weight,
                label_smoothing=self.label_smoothing,
            )
            total_loss = total_loss + loss

        return total_loss / self.num_emotions


class FocalCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        num_emotions: int = 4,
        gamma: float = 2.0,
        alpha: Optional[List[List[float]]] = None,
    ):
        super().__init__()
        self.num_emotions = num_emotions
        self.gamma = gamma

        if alpha is not None:
            self.alpha = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.tensor(a, dtype=torch.float32), requires_grad=False
                    )
                    for a in alpha
                ]
            )
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss: torch.Tensor = torch.tensor(0.0, device=logits.device)
        for i in range(self.num_emotions):
            emotion_logits = logits[:, i, :]
            emotion_targets = targets[:, i]

            ce_loss = F.cross_entropy(emotion_logits, emotion_targets, reduction="none")
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** self.gamma

            if self.alpha is not None:
                alpha_t = self.alpha[i][emotion_targets]
                focal_loss = alpha_t * focal_weight * ce_loss
            else:
                focal_loss = focal_weight * ce_loss

            total_loss = total_loss + focal_loss.mean()

        return total_loss / self.num_emotions


class LabelSmoothingMSE(nn.Module):
    def __init__(self, smoothing: float = 0.1, num_classes: int = 4):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smoothed = targets * (1 - self.smoothing) + (
            self.smoothing / self.num_classes
        )
        return F.mse_loss(predictions, targets_smoothed)


class OrdinalRegressionLoss(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size, num_emotions = predictions.shape

        cumulative_probs = torch.zeros(
            batch_size, num_emotions, self.num_classes - 1, device=predictions.device
        )
        for i in range(self.num_classes - 1):
            threshold = (i + 1) / self.num_classes
            cumulative_probs[:, :, i] = torch.sigmoid((predictions - threshold) * 10)

        loss: torch.Tensor = torch.tensor(0.0, device=predictions.device)
        for i in range(self.num_classes - 1):
            target_cumulative = (targets > i).float()
            loss = loss + F.binary_cross_entropy(cumulative_probs[:, :, i], target_cumulative)

        return loss / (self.num_classes - 1)


class WeightedMSELoss(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.class_weights = class_weights

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse = (predictions - targets) ** 2

        if self.class_weights is not None:
            target_classes = torch.round(
                torch.clamp(targets, 0, len(self.class_weights) - 1)
            ).long()
            weights = self.class_weights[target_classes]
            mse = mse * weights

        return mse.mean()


class FocalMSELoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse = (predictions - targets) ** 2
        focal_weight = (1 - torch.exp(-mse)) ** self.gamma
        return (focal_weight * mse).mean()


class CombinedLoss(nn.Module):
    def __init__(
        self, mse_weight: float = 1.0, ordinal_weight: float = 0.0, num_classes: int = 4
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.ordinal_weight = ordinal_weight
        self.mse_loss = nn.MSELoss()
        self.ordinal_loss = OrdinalRegressionLoss(num_classes)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = torch.tensor(0.0, device=predictions.device)

        if self.mse_weight > 0:
            loss = loss + self.mse_weight * self.mse_loss(predictions, targets)

        if self.ordinal_weight > 0:
            loss = loss + self.ordinal_weight * self.ordinal_loss(predictions, targets)

        return loss


def create_loss_function(config: dict) -> nn.Module:
    loss_config = config.get("loss", {})
    loss_type = loss_config.get("type", "mse").lower()

    if loss_type == "mse":
        return nn.MSELoss()

    elif loss_type == "mae":
        return nn.L1Loss()

    elif loss_type == "smooth_l1":
        return nn.SmoothL1Loss()

    elif loss_type == "label_smoothing_mse":
        smoothing = loss_config.get("smoothing", 0.1)
        num_classes = loss_config.get("num_classes", 4)
        return LabelSmoothingMSE(smoothing, num_classes)

    elif loss_type == "ordinal":
        num_classes = loss_config.get("num_classes", 4)
        return OrdinalRegressionLoss(num_classes)

    elif loss_type == "weighted_mse":
        class_weights = loss_config.get("class_weights", None)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        return WeightedMSELoss(class_weights)

    elif loss_type == "focal_mse":
        gamma = loss_config.get("gamma", 2.0)
        return FocalMSELoss(gamma)

    elif loss_type == "combined":
        mse_weight = loss_config.get("mse_weight", 1.0)
        ordinal_weight = loss_config.get("ordinal_weight", 0.5)
        num_classes = loss_config.get("num_classes", 4)
        return CombinedLoss(mse_weight, ordinal_weight, num_classes)

    elif loss_type == "cross_entropy":
        num_emotions = loss_config.get("num_emotions", 4)
        class_weights = loss_config.get("class_weights", None)
        label_smoothing = loss_config.get("label_smoothing", 0.0)
        return MultiHeadCrossEntropyLoss(num_emotions, class_weights, label_smoothing)

    elif loss_type == "focal_ce":
        num_emotions = loss_config.get("num_emotions", 4)
        gamma = loss_config.get("gamma", 2.0)
        alpha = loss_config.get("alpha", None)
        return FocalCrossEntropyLoss(num_emotions, gamma, alpha)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
