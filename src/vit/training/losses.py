import pandas as pd
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
        # logits: (batch, num_emotions, num_classes)
        # targets: (batch, num_emotions) as torch.long
        total_loss = torch.tensor(0.0, device=logits.device)
        for i in range(self.num_emotions):
            weight = (
                self.class_weights[i].to(logits.device)
                if self.class_weights is not None
                else None
            )
            loss = F.cross_entropy(
                logits[:, i, :],
                targets[:, i],
                weight=weight,
                label_smoothing=self.label_smoothing,
            )
            total_loss = total_loss + loss
        return total_loss / self.num_emotions


class MultiHeadFocalLoss(nn.Module):
    def __init__(
        self,
        num_emotions: int = 4,
        gamma: float = 2.0,
        class_weights: Optional[List[List[float]]] = None,
    ):
        super().__init__()
        self.num_emotions = num_emotions
        self.gamma = gamma

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
        total_loss = torch.tensor(0.0, device=logits.device)
        for i in range(self.num_emotions):
            weight = (
                self.class_weights[i].to(logits.device)
                if self.class_weights is not None
                else None
            )
            ce = F.cross_entropy(
                logits[:, i, :], targets[:, i], weight=weight, reduction="none"
            )
            pt = torch.exp(-ce)
            focal = ((1 - pt) ** self.gamma) * ce
            total_loss = total_loss + focal.mean()
        return total_loss / self.num_emotions


class MultiHeadBCELoss(nn.Module):
    def __init__(
        self,
        num_emotions: int = 4,
        pos_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_emotions = num_emotions

        if pos_weights is not None:
            self.pos_weights = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.tensor([w], dtype=torch.float32), requires_grad=False
                    )
                    for w in pos_weights
                ]
            )
        else:
            self.pos_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Use the difference of class-1 and class-0 logits as the binary score.
        total_loss = torch.tensor(0.0, device=logits.device)
        for i in range(self.num_emotions):
            score = logits[:, i, 1] - logits[:, i, 0]
            pos_weight = (
                self.pos_weights[i].to(logits.device)
                if self.pos_weights is not None
                else None
            )
            loss = F.binary_cross_entropy_with_logits(
                score,
                targets[:, i].float(),
                pos_weight=pos_weight,
            )
            total_loss = total_loss + loss
        return total_loss / self.num_emotions


class MultiHeadAsymmetricLoss(nn.Module):
    def __init__(
        self,
        num_emotions: int = 4,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
    ):
        super().__init__()
        self.num_emotions = num_emotions
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Force float32: log/sigmoid are numerically unstable in float16 (AMP),
        # where clamp(min=1e-8) is effectively 0 and log(0) = -inf → NaN.
        logits = logits.float()

        total_loss = torch.tensor(0.0, device=logits.device)
        for i in range(self.num_emotions):
            score = logits[:, i, 1] - logits[:, i, 0]
            prob = torch.sigmoid(score)
            t = targets[:, i].float()

            # Clip negatives: shift probability margin to reduce trivially easy negatives
            prob_neg = prob
            if self.clip > 0:
                prob_neg = (prob + self.clip).clamp(max=1.0)

            loss_pos = t * torch.log(prob.clamp(min=1e-6))
            loss_neg = (1 - t) * torch.log((1 - prob_neg).clamp(min=1e-6))

            # Apply asymmetric focusing
            if self.gamma_pos > 0:
                loss_pos = loss_pos * ((1 - prob) ** self.gamma_pos)
            if self.gamma_neg > 0:
                loss_neg = loss_neg * (prob_neg**self.gamma_neg)

            loss = -(loss_pos + loss_neg).mean()
            total_loss = total_loss + loss

        return total_loss / self.num_emotions


def _compute_class_weights(
    train_csv: str, emotion_columns: List[str], num_classes: int
) -> List[List[float]]:
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


def _compute_pos_weights(train_csv: str, emotion_columns: List[str]) -> List[float]:
    df = pd.read_csv(train_csv)
    pos_weights = []
    for col in emotion_columns:
        neg = (df[col] == 0).sum()
        pos = (df[col] == 1).sum()
        w = float(neg / pos) if pos > 0 else 1.0
        pos_weights.append(w)
        print(f"  {col}: pos_weight={w:.3f}  (neg={neg}, pos={pos})")
    return pos_weights


class MultiHeadHammingLoss(nn.Module):
    def __init__(self, num_emotions: int = 4):
        super().__init__()
        self.num_emotions = num_emotions

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (batch, num_emotions, num_classes)
        # targets: (batch, num_emotions) as torch.long
        # Soft Hamming: 1 - p(correct_class) per emotion head, averaged.
        # Differentiable via softmax; approximates Hamming distance:
        # ~0 when confident and correct, ~1 when wrong.
        total_loss = torch.tensor(0.0, device=logits.device)
        for i in range(self.num_emotions):
            probs = torch.softmax(logits[:, i, :], dim=-1)  # (batch, num_classes)
            correct_prob = probs.gather(1, targets[:, i].unsqueeze(1)).squeeze(1)  # (batch,)
            total_loss = total_loss + (1.0 - correct_prob).mean()
        return total_loss / self.num_emotions


class ExactMatchWrapper(nn.Module):
    def __init__(self, base_loss: nn.Module, exact_match_weight: float = 0.2):
        super().__init__()
        self.base_loss = base_loss
        self.exact_match_weight = exact_match_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.base_loss(logits, targets)
        preds = logits.argmax(dim=2)
        exact = (preds == targets).all(dim=1).float()
        exact_loss = 1.0 - exact.mean()
        return loss + self.exact_match_weight * exact_loss


EMOTION_COLUMNS = ["Boredom", "Engagement", "Confusion", "Frustration"]


def create_loss_function(config: dict) -> nn.Module:
    loss_config = config.get("loss", {})
    loss_type = loss_config.get("type", "cross_entropy").lower()
    num_emotions = config["model"]["num_emotions"]
    num_classes = config["model"].get("num_classes", 4)

    if loss_type == "hamming":
        print("Using Hamming Loss")
        return MultiHeadHammingLoss(num_emotions)

    if loss_type == "focal":
        gamma = loss_config.get("gamma", 2.0)
        raw_weights = loss_config.get("class_weights", None)
        if raw_weights == "auto":
            train_csv = config["data"]["train_csv"]
            print("Computing class weights from training data...")
            raw_weights = _compute_class_weights(
                train_csv, EMOTION_COLUMNS, num_classes
            )
        print(f"Using Focal Loss (gamma={gamma})")
        return _maybe_wrap_exact_match(
            MultiHeadFocalLoss(num_emotions, gamma=gamma, class_weights=raw_weights),
            loss_config,
        )

    if loss_type == "bce":
        pos_weights = loss_config.get("pos_weight", None)
        if pos_weights == "auto":
            train_csv = config["data"]["train_csv"]
            print("Computing positive class weights from training data...")
            pos_weights = _compute_pos_weights(train_csv, EMOTION_COLUMNS)
        print("Using BCE with Logits Loss")
        return _maybe_wrap_exact_match(
            MultiHeadBCELoss(num_emotions, pos_weights=pos_weights), loss_config
        )

    if loss_type == "asymmetric":
        gamma_pos = loss_config.get("gamma_pos", 0.0)
        gamma_neg = loss_config.get("gamma_neg", 4.0)
        clip = loss_config.get("clip", 0.05)
        print(
            f"Using Asymmetric Loss (gamma_pos={gamma_pos}, gamma_neg={gamma_neg}, clip={clip})"
        )
        return _maybe_wrap_exact_match(
            MultiHeadAsymmetricLoss(
                num_emotions, gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip
            ),
            loss_config,
        )

    # Default: cross_entropy
    label_smoothing = loss_config.get("label_smoothing", 0.0)
    raw_weights = loss_config.get("class_weights", None)
    if raw_weights == "auto":
        train_csv = config["data"]["train_csv"]
        print("Computing class weights from training data...")
        raw_weights = _compute_class_weights(train_csv, EMOTION_COLUMNS, num_classes)
    base = MultiHeadCrossEntropyLoss(num_emotions, raw_weights, label_smoothing)
    return _maybe_wrap_exact_match(base, loss_config)


def _maybe_wrap_exact_match(base: nn.Module, loss_config: dict) -> nn.Module:
    weight = loss_config.get("exact_match_weight", 0.0)
    if weight > 0.0:
        print(f"Exact match auxiliary loss enabled (weight={weight})")
        return ExactMatchWrapper(base, exact_match_weight=weight)
    return base
