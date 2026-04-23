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


class MultiHeadFocalLoss(nn.Module):
    """Focal loss averaged across all emotion heads.

    Down-weights easy (well-classified) examples so training focuses on hard
    minority-class samples. Particularly useful when class imbalance is moderate
    (e.g. Frustration 78/22, Confusion 69/31).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

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
            self.class_weights = nn.ParameterList([
                nn.Parameter(torch.tensor(w, dtype=torch.float32), requires_grad=False)
                for w in class_weights
            ])
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=logits.device)
        for i in range(self.num_emotions):
            weight = self.class_weights[i].to(logits.device) if self.class_weights is not None else None
            ce = F.cross_entropy(logits[:, i, :], targets[:, i], weight=weight, reduction='none')
            pt = torch.exp(-ce)
            focal = ((1 - pt) ** self.gamma) * ce
            total_loss = total_loss + focal.mean()
        return total_loss / self.num_emotions


class MultiHeadBCELoss(nn.Module):
    """Binary cross-entropy (sigmoid) loss averaged across all emotion heads.

    Uses BCEWithLogitsLoss instead of 2-class softmax. Each class is treated
    independently (sigmoid per logit) rather than competing (softmax), which
    gives cleaner gradients for binary tasks.

    Expects logits shape (batch, num_emotions, 2); uses the class-1 logit
    (logits[:, i, 1] - logits[:, i, 0]) as the signed score.

    pos_weight per emotion balances the positive class when it is the minority.
    """

    def __init__(
        self,
        num_emotions: int = 4,
        pos_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.num_emotions = num_emotions

        if pos_weights is not None:
            self.pos_weights = nn.ParameterList([
                nn.Parameter(torch.tensor([w], dtype=torch.float32), requires_grad=False)
                for w in pos_weights
            ])
        else:
            self.pos_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Use the difference of class-1 and class-0 logits as the binary score.
        total_loss = torch.tensor(0.0, device=logits.device)
        for i in range(self.num_emotions):
            score = logits[:, i, 1] - logits[:, i, 0]
            pos_weight = self.pos_weights[i].to(logits.device) if self.pos_weights is not None else None
            loss = F.binary_cross_entropy_with_logits(
                score,
                targets[:, i].float(),
                pos_weight=pos_weight,
            )
            total_loss = total_loss + loss
        return total_loss / self.num_emotions


class MultiHeadAsymmetricLoss(nn.Module):
    """Asymmetric loss (ASL) averaged across all emotion heads.

    Designed for multi-label imbalanced binary classification. Applies separate
    focusing parameters for positives (gamma_pos) and negatives (gamma_neg),
    aggressively down-weighting easy negatives (the dominant class 0) while
    keeping sensitivity to true positives.

    Reference: Ben-Baruch et al. (2021) "Asymmetric Loss For Multi-Label Classification"

    gamma_pos: focusing for positive samples (default 0 — no focusing on positives)
    gamma_neg: focusing for negative samples (default 4 — strong down-weighting of easy negatives)
    clip:      probability margin to shift negatives away from 0 (default 0.05)
    """

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
        total_loss = torch.tensor(0.0, device=logits.device)
        for i in range(self.num_emotions):
            score = logits[:, i, 1] - logits[:, i, 0]
            prob = torch.sigmoid(score)
            t = targets[:, i].float()

            # Clip negatives: shift probability margin to reduce trivially easy negatives
            prob_neg = prob
            if self.clip > 0:
                prob_neg = (prob + self.clip).clamp(max=1.0)

            loss_pos = t * torch.log(prob.clamp(min=1e-8))
            loss_neg = (1 - t) * torch.log((1 - prob_neg).clamp(min=1e-8))

            # Apply asymmetric focusing
            if self.gamma_pos > 0:
                loss_pos = loss_pos * ((1 - prob) ** self.gamma_pos)
            if self.gamma_neg > 0:
                loss_neg = loss_neg * (prob_neg ** self.gamma_neg)

            loss = -(loss_pos + loss_neg).mean()
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


def _compute_pos_weights(train_csv: str, emotion_columns: List[str]) -> List[float]:
    """Compute per-emotion positive class weight for BCE: count(neg) / count(pos)."""
    df = pd.read_csv(train_csv)
    pos_weights = []
    for col in emotion_columns:
        neg = (df[col] == 0).sum()
        pos = (df[col] == 1).sum()
        w = float(neg / pos) if pos > 0 else 1.0
        pos_weights.append(w)
        print(f"  {col}: pos_weight={w:.3f}  (neg={neg}, pos={pos})")
    return pos_weights


EMOTION_COLUMNS = ["Boredom", "Engagement", "Confusion", "Frustration"]


def create_loss_function(config: dict) -> nn.Module:
    loss_config = config.get('loss', {})
    loss_type = loss_config.get('type', 'cross_entropy').lower()
    num_emotions = config['model']['num_emotions']
    num_classes = config['model'].get('num_classes', 4)

    if loss_type == 'focal':
        gamma = loss_config.get('gamma', 2.0)
        raw_weights = loss_config.get('class_weights', None)
        if raw_weights == 'auto':
            train_csv = config['data']['train_csv']
            print("Computing class weights from training data...")
            raw_weights = _compute_class_weights(train_csv, EMOTION_COLUMNS, num_classes)
        print(f"Using Focal Loss (gamma={gamma})")
        return MultiHeadFocalLoss(num_emotions, gamma=gamma, class_weights=raw_weights)

    if loss_type == 'bce':
        pos_weights = loss_config.get('pos_weight', None)
        if pos_weights == 'auto':
            train_csv = config['data']['train_csv']
            print("Computing positive class weights from training data...")
            pos_weights = _compute_pos_weights(train_csv, EMOTION_COLUMNS)
        print("Using BCE with Logits Loss")
        return MultiHeadBCELoss(num_emotions, pos_weights=pos_weights)

    if loss_type == 'asymmetric':
        gamma_pos = loss_config.get('gamma_pos', 0.0)
        gamma_neg = loss_config.get('gamma_neg', 4.0)
        clip = loss_config.get('clip', 0.05)
        print(f"Using Asymmetric Loss (gamma_pos={gamma_pos}, gamma_neg={gamma_neg}, clip={clip})")
        return MultiHeadAsymmetricLoss(num_emotions, gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)

    # Default: cross_entropy
    label_smoothing = loss_config.get('label_smoothing', 0.0)
    raw_weights = loss_config.get('class_weights', None)
    if raw_weights == 'auto':
        train_csv = config['data']['train_csv']
        print("Computing class weights from training data...")
        raw_weights = _compute_class_weights(train_csv, EMOTION_COLUMNS, num_classes)
    return MultiHeadCrossEntropyLoss(num_emotions, raw_weights, label_smoothing)
