import argparse
from pathlib import Path
from typing import Dict, Any
import yaml
import os
from dotenv import load_dotenv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from ..data.dataloader import create_dataloaders, get_dataset_info
from ..models.landmark_vit import create_model
from .utils import (
    save_checkpoint,
    create_optimizer,
    create_scheduler,
    calculate_metrics,
    EarlyStopping
)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Dict[str, Any]
) -> float:
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    for landmarks, labels in progress_bar:
        landmarks = landmarks.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(landmarks)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        if config['training']['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['gradient_clip']
            )
        
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> tuple[float, Dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        for landmarks, labels in progress_bar:
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            
            outputs = model(landmarks)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_predictions.append(outputs)
            all_targets.append(labels)
            
            progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(val_loader)
    
    all_predictions_tensor = torch.cat(all_predictions, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_predictions_tensor, all_targets_tensor)
    
    return avg_loss, metrics


def train(config: Dict[str, Any]) -> None:
    device_config = config['device']
    if device_config['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_config['cuda_device']}")
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader, _ = create_dataloaders(config)
    dataset_info = get_dataset_info(config)
    print(f"Train: {dataset_info['train_size']}, Val: {dataset_info['val_size']}, Test: {dataset_info['test_size']}")
    print(f"Emotions: {dataset_info['emotion_columns']}")
    
    model = create_model(config)
    model = model.to(device)
    print(f"Trainable params: {model.get_num_trainable_params():,}")
    
    criterion = nn.MSELoss()
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    early_stopping = EarlyStopping(
        patience=config['early_stopping']['patience'],
        min_delta=config['early_stopping']['min_delta']
    )
    
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(config['training']['num_epochs']):
        if config['model']['gradual_unfreeze']['enabled']:
            unfreeze_epoch = config['model']['gradual_unfreeze']['unfreeze_at_epoch']
            if epoch == unfreeze_epoch:
                print(f"\n{'='*60}")
                print(f"UNFREEZING BACKBONE at epoch {epoch+1}")
                print(f"{'='*60}")
                model.unfreeze_backbone()
                
                if config['model']['gradual_unfreeze']['reduce_lr_on_unfreeze']:
                    reduction_factor = config['model']['gradual_unfreeze']['lr_reduction_factor']
                    old_lr = optimizer.param_groups[0]['lr']
                    new_lr = old_lr * reduction_factor
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"Reduced learning rate: {old_lr:.6f} -> {new_lr:.6f}")
                
                trainable_params = model.get_num_trainable_params()
                print(f"Trainable parameters: {trainable_params:,}")
                print(f"{'='*60}\n")
        
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )
        
        metrics_to_log = {
            'epoch': epoch,
            'train/loss': train_loss,
            'val/loss': val_loss,
            'val/mse': val_metrics['mse'],
            'val/mae': val_metrics['mae'],
            'val/rmse': val_metrics['rmse'],
            'learning_rate': optimizer.param_groups[0]['lr'],
        }
        
        for i, emotion in enumerate(dataset_info['emotion_columns']):
            metrics_to_log[f'val/mse_{emotion}'] = val_metrics['mse_per_emotion'][i]
            metrics_to_log[f'val/mae_{emotion}'] = val_metrics['mae_per_emotion'][i]
            metrics_to_log[f'val/corr_{emotion}'] = val_metrics['correlation_per_emotion'][i]
        
        wandb.log(metrics_to_log)
        
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val MSE: {val_metrics['mse']:.4f}, MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
        
        if config['scheduler']['type'].lower() == 'reduce_on_plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_loss, val_loss, str(checkpoint_path),
                best_val_loss, config
            )
            print(f"Saved best model: val_loss={val_loss:.4f}")
        
        if not config['output']['save_best_only'] and (epoch + 1) % config['output']['save_frequency'] == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_loss, val_loss, str(checkpoint_path),
                best_val_loss, config
            )
        
        if early_stopping(val_loss):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    wandb.finish()
    print(f"\nTraining completed! Best val loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    import sys
    print("Please use scripts/train_landmark.py to run training")
    sys.exit(1)
