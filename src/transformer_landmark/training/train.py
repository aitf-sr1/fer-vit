from pathlib import Path
from typing import Dict, Any
import os
from datetime import datetime
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from ..data.dataloader import create_dataloaders, get_dataset_info
from ..models import create_model
from .losses import create_loss_function
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
    epoch: int,
    mode: str = "classification"
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
    metrics = calculate_metrics(all_predictions_tensor, all_targets_tensor, mode=mode)

    if mode == "classification":
        num_emotions = all_predictions_tensor.shape[1]
        per_emotion_losses: dict[str, float] = {}
        for i in range(num_emotions):
            per_emotion_losses[f"emotion_{i}"] = float(
                F.cross_entropy(all_predictions_tensor[:, i, :], all_targets_tensor[:, i])
            )
        metrics['per_emotion_losses'] = per_emotion_losses
        metrics['all_preds'] = all_predictions_tensor.argmax(dim=-1).cpu().numpy()
        metrics['all_targets'] = all_targets_tensor.cpu().numpy()

    return avg_loss, metrics


def train(config: Dict[str, Any]) -> None:
    load_dotenv()
    
    wandb_config = {
        'mode': os.getenv('WANDB_MODE', 'online'),
        'entity': os.getenv('WANDB_ENTITY'),
        'project': os.getenv('WANDB_PROJECT', 'fer-vit'),
    }
    
    if wandb_config['mode'] == 'disabled':
        print("⚠️  wandb logging is DISABLED (WANDB_MODE=disabled)")
    
    model_name = config['model']['name']
    wandb.init(
        project=wandb_config['project'],
        entity=wandb_config['entity'],
        mode=wandb_config['mode'],
        config=config,
        name=config.get('experiment_name', f'landmark-{model_name}'),
        tags=['landmark', model_name, 'emotion-detection']
    )
    
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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config['model']['name']}")
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")
    
    wandb.watch(model, log='all', log_freq=100)
    
    criterion = create_loss_function(config)
    criterion = criterion.to(device)
    print(f"Loss function: {config['loss']['type']}")
    
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    early_stopping = EarlyStopping(
        patience=config['early_stopping']['patience'],
        min_delta=config['early_stopping']['min_delta']
    )
    
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_checkpoint_dir = checkpoint_dir / timestamp
    run_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    latest_link = checkpoint_dir / 'latest'
    if latest_link.exists() and latest_link.is_symlink():
        latest_link.unlink()
    if latest_link.exists():
        import shutil
        shutil.rmtree(latest_link)
    try:
        latest_link.symlink_to(timestamp, target_is_directory=True)
    except OSError:
        pass
    
    print(f"Checkpoint directory: {run_checkpoint_dir}")
    
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    model_mode = config.get('model', {}).get('mode', 'classification')
    
    for epoch in range(config['training']['num_epochs']):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch, mode=model_mode
        )
        
        metrics_to_log = {
            'epoch': epoch,
            'train/loss': train_loss,
            'val/loss': val_loss,
            'val/mae': val_metrics['mae'],
            'learning_rate': optimizer.param_groups[0]['lr'],
        }
        
        if model_mode == "classification":
            metrics_to_log['val/accuracy'] = val_metrics['accuracy']
            metrics_to_log['val/exact_match'] = val_metrics['exact_match']
            emotions = dataset_info['emotion_columns']
            num_classes = config['model']['num_classes']
            class_names_per_emotion = [
                [f'class_{c}' for c in range(num_classes)]
                for _ in range(len(emotions))
            ]
            all_preds = val_metrics['all_preds']
            all_targets_np = val_metrics['all_targets']
            for i, emotion in enumerate(emotions):
                metrics_to_log[f'val/acc_{emotion}'] = val_metrics['accuracy_per_emotion'][i]
                metrics_to_log[f'val/mae_{emotion}'] = val_metrics['mae_per_emotion'][i]
                metrics_to_log[f'val/kappa_{emotion}'] = val_metrics['kappa_per_emotion'][i]
                metrics_to_log[f'val/loss_{emotion}'] = val_metrics['per_emotion_losses'][f'emotion_{i}']

                per_class = val_metrics['per_class_accuracy'][i]
                for cls_idx, cls_acc in enumerate(per_class):
                    cls_name = class_names_per_emotion[i][cls_idx]
                    metrics_to_log[f'val/cls_acc_{emotion}_{cls_name}'] = cls_acc

                preds_i = all_preds[:, i].tolist()
                targets_i = all_targets_np[:, i].tolist()
                cls_names = class_names_per_emotion[i]
                metrics_to_log[f'val/confusion_matrix_{emotion}'] = wandb.plot.confusion_matrix(
                    y_true=targets_i, preds=preds_i, class_names=cls_names
                )
                metrics_to_log[f'val/pred_dist_{emotion}'] = wandb.Histogram(preds_i)

            if device.type == 'cuda':
                metrics_to_log['gpu/memory_allocated_mb'] = torch.cuda.memory_allocated(device) / 1e6
                metrics_to_log['gpu/memory_reserved_mb'] = torch.cuda.memory_reserved(device) / 1e6
        else:
            metrics_to_log['val/mse'] = val_metrics['mse']
            metrics_to_log['val/rmse'] = val_metrics['rmse']
            for i, emotion in enumerate(dataset_info['emotion_columns']):
                metrics_to_log[f'val/mse_{emotion}'] = val_metrics['mse_per_emotion'][i]
                metrics_to_log[f'val/mae_{emotion}'] = val_metrics['mae_per_emotion'][i]
                metrics_to_log[f'val/corr_{emotion}'] = val_metrics['correlation_per_emotion'][i]
        
        wandb.log(metrics_to_log)
        
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if model_mode == "classification":
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}, Exact Match: {val_metrics['exact_match']:.4f}, MAE: {val_metrics['mae']:.4f}")
        else:
            print(f"Val MSE: {val_metrics['mse']:.4f}, MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
        
        if config['scheduler']['type'].lower() == 'reduce_on_plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = run_checkpoint_dir / f'best_model_epoch_{epoch+1}_loss_{val_loss:.4f}.pth'
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_loss, val_loss, str(checkpoint_path),
                best_val_loss, config
            )
            print(f"Saved best model: epoch={epoch+1}, val_loss={val_loss:.4f}")
        
        if not config['output']['save_best_only'] and (epoch + 1) % config['output']['save_frequency'] == 0:
            checkpoint_path = run_checkpoint_dir / f'checkpoint_epoch_{epoch+1}_loss_{val_loss:.4f}.pth'
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
