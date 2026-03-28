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
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler


def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    mse_per_emotion = np.mean((predictions_np - targets_np) ** 2, axis=0)
    mae_per_emotion = np.mean(np.abs(predictions_np - targets_np), axis=0)
    
    overall_mse = float(np.mean(mse_per_emotion))
    overall_mae = float(np.mean(mae_per_emotion))
    overall_rmse = float(np.sqrt(overall_mse))
    
    correlations = []
    for i in range(predictions_np.shape[1]):
        if predictions_np[:, i].std() > 0 and targets_np[:, i].std() > 0:
            corr = np.corrcoef(predictions_np[:, i], targets_np[:, i])[0, 1]
            correlations.append(corr)
        else:
            correlations.append(0.0)
    
    metrics = {
        'mse': overall_mse,
        'mae': overall_mae,
        'rmse': overall_rmse,
        'mse_per_emotion': mse_per_emotion.tolist(),
        'mae_per_emotion': mae_per_emotion.tolist(),
        'correlation_per_emotion': correlations
    }
    
    return metrics


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
