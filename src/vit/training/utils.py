from typing import Dict, Any, Optional
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    epoch: int,
    train_loss: float,
    val_loss: float,
    checkpoint_path: str,
    best_val_loss: float,
    config: Dict[str, Any]
) -> None:
    checkpoint_dir = Path(checkpoint_path).parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'config': config
    }
    
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
    optimizer_type = config['optimizer']['type'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=config['optimizer']['betas'],
            eps=config['optimizer']['eps'],
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=config['optimizer']['betas'],
            eps=config['optimizer']['eps'],
            weight_decay=weight_decay
        )
    elif optimizer_type == 'rmsprop':
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=lr,
            momentum=config['optimizer'].get('momentum', 0.0),
            alpha=config['optimizer'].get('alpha', 0.99),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: Dict[str, Any]) -> Any:
    scheduler_type = config['scheduler']['type'].lower()
    
    if scheduler_type == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config['scheduler']['mode'],
            factor=config['scheduler']['factor'],
            patience=config['scheduler']['patience'],
            min_lr=config['scheduler']['min_lr']
        )
    elif scheduler_type == 'cosine_annealing_warm_restarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['scheduler']['T_0'],
            T_mult=config['scheduler']['T_mult'],
            eta_min=config['scheduler']['eta_min']
        )
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    elif scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
    elif scheduler_type == 'one_cycle':
        # Requires steps_per_epoch and num_epochs injected by train(); see train.py.
        steps_per_epoch = config['_one_cycle_steps_per_epoch']
        num_epochs = config['training']['num_epochs']
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['scheduler']['max_lr'],
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            pct_start=config['scheduler'].get('pct_start', 0.3),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, Any]:
    return _classification_metrics(predictions, targets)


def _cohen_kappa(preds: np.ndarray, targets: np.ndarray) -> float:
    n = len(preds)
    if n == 0:
        return 0.0
    num_classes = int(max(preds.max(), targets.max())) + 1
    p_o = float((preds == targets).mean())
    pred_freq = np.bincount(preds.astype(int), minlength=num_classes) / n
    true_freq = np.bincount(targets.astype(int), minlength=num_classes) / n
    p_e = float(np.dot(pred_freq, true_freq))
    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def _classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
    # logits: (batch, num_emotions, num_classes), targets: (batch, num_emotions) long
    num_classes = logits.shape[2]
    preds_np = logits.argmax(dim=2).detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    num_emotions = preds_np.shape[1]

    accuracy_per_emotion = [
        float((preds_np[:, i] == targets_np[:, i]).mean())
        for i in range(num_emotions)
    ]
    mae_per_emotion = [
        float(np.abs(preds_np[:, i] - targets_np[:, i]).mean())
        for i in range(num_emotions)
    ]
    overall_accuracy = float(np.mean(accuracy_per_emotion))
    overall_mae = float(np.mean(mae_per_emotion))
    exact_match = float((preds_np == targets_np).all(axis=1).mean())

    kappa_per_emotion = [
        _cohen_kappa(preds_np[:, i], targets_np[:, i])
        for i in range(num_emotions)
    ]

    per_class_accuracy = []
    for i in range(num_emotions):
        class_accs = []
        for c in range(num_classes):
            mask = targets_np[:, i] == c
            if mask.sum() > 0:
                class_accs.append(float((preds_np[mask, i] == c).mean()))
            else:
                class_accs.append(float('nan'))
        per_class_accuracy.append(class_accs)

    result: Dict[str, Any] = {
        'accuracy': overall_accuracy,
        'exact_match': exact_match,
        'mae': overall_mae,
        'accuracy_per_emotion': accuracy_per_emotion,
        'mae_per_emotion': mae_per_emotion,
        'kappa_per_emotion': kappa_per_emotion,
        'per_class_accuracy': per_class_accuracy,
    }

    if num_classes == 2:
        precision_per_emotion, recall_per_emotion, f1_per_emotion = [], [], []
        for i in range(num_emotions):
            p, r = preds_np[:, i], targets_np[:, i]
            tp = float(((p == 1) & (r == 1)).sum())
            fp = float(((p == 1) & (r == 0)).sum())
            fn = float(((p == 0) & (r == 1)).sum())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            precision_per_emotion.append(precision)
            recall_per_emotion.append(recall)
            f1_per_emotion.append(f1)

        result['precision'] = float(np.mean(precision_per_emotion))
        result['recall'] = float(np.mean(recall_per_emotion))
        result['f1'] = float(np.mean(f1_per_emotion))
        result['precision_per_emotion'] = precision_per_emotion
        result['recall_per_emotion'] = recall_per_emotion
        result['f1_per_emotion'] = f1_per_emotion

    return result


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop
