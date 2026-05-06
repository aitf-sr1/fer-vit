from pathlib import Path
from typing import Dict, Any
import os
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from ..data.dataloader import create_dataloaders, get_dataset_info
from ..models.vit_model import create_model
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
    config: Dict[str, Any],
    scheduler=None,
    scaler=None,
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    grad_norms: list[float] = []
    is_one_cycle = config['scheduler']['type'].lower() == 'one_cycle'
    use_amp = scaler is not None
    grad_accum_steps = config['training'].get('gradient_accumulation_steps', 1)

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
    for step, (images, labels) in enumerate(progress_bar):
        images = images.to(device)
        labels = labels.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if config['training']['gradient_clip'] > 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config['training']['gradient_clip']
                ))
            else:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float('inf')
                ))
            grad_norms.append(grad_norm)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

            if is_one_cycle and scheduler is not None:
                scheduler.step()

        loss_val = loss.item() * grad_accum_steps
        if not torch.isnan(loss).item():
            total_loss += loss_val
        progress_bar.set_postfix({'loss': loss_val})

    avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
    max_grad_norm = float(np.max(grad_norms)) if grad_norms else 0.0
    return total_loss / len(train_loader), avg_grad_norm, max_grad_norm


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_amp: bool = False,
) -> tuple[float, Dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_predictions.append(outputs.float())
            all_targets.append(labels)

            progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(val_loader)
    all_predictions_tensor = torch.cat(all_predictions, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_predictions_tensor, all_targets_tensor)

    # Per-emotion CE loss: outputs shape (N, num_emotions, num_classes)
    num_emotions = all_predictions_tensor.shape[1]
    per_emotion_losses: dict[str, float] = {}
    for i in range(num_emotions):
        per_emotion_losses[f"emotion_{i}"] = float(
            F.cross_entropy(all_predictions_tensor[:, i, :], all_targets_tensor[:, i])
        )
    metrics['per_emotion_losses'] = per_emotion_losses

    # Raw preds/targets as numpy for confusion matrix and distribution logging
    metrics['all_preds'] = (
        all_predictions_tensor.argmax(dim=-1).cpu().numpy()
    )  # shape (N, num_emotions)
    metrics['all_targets'] = all_targets_tensor.cpu().numpy()

    return avg_loss, metrics


def _log_prediction_table(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    emotion_columns: list[str],
    config: Dict[str, Any],
    n_samples: int = 16,
) -> None:
    norm_cfg = config['augmentation']['imagenet_norm']
    mean = torch.tensor(norm_cfg['mean'], device=device).view(3, 1, 1)
    std = torch.tensor(norm_cfg['std'], device=device).view(3, 1, 1)
    class_names_per_emotion = config['data']['class_names']

    columns = ['image'] + [f'true_{e}' for e in emotion_columns] + [f'pred_{e}' for e in emotion_columns]
    table = wandb.Table(columns=columns)

    model.eval()
    collected = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=-1)  # (B, num_emotions)

            for b in range(images.shape[0]):
                if collected >= n_samples:
                    break
                img = images[b] * std + mean
                img = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                img = (img * 255).astype(np.uint8)
                wandb_img = wandb.Image(img)
                true_labels = [class_names_per_emotion[i][labels[b, i].item()] for i in range(len(emotion_columns))]
                pred_labels = [class_names_per_emotion[i][preds[b, i].item()] for i in range(len(emotion_columns))]
                table.add_data(wandb_img, *true_labels, *pred_labels)
                collected += 1

            if collected >= n_samples:
                break

    wandb.log({'val/prediction_samples': table})



    load_dotenv()

    wandb_config = {
        'mode': os.getenv('WANDB_MODE', 'online'),
        'entity': os.getenv('WANDB_ENTITY'),
        'project': os.getenv('WANDB_PROJECT', 'fer-vit'),
    }

    if wandb_config['mode'] == 'disabled':
        print("wandb logging is DISABLED (WANDB_MODE=disabled)")

    model_name = config['model']['name']
    base_tags = ['vit', model_name, 'emotion-detection']
    extra_tags = config.get('tags', [])
    all_tags = base_tags + [t for t in extra_tags if t not in base_tags]
    wandb.init(
        project=wandb_config['project'],
        entity=wandb_config['entity'],
        mode=wandb_config['mode'],
        config=config,
        name=config.get('experiment_name', f'vit-{model_name}'),
        tags=all_tags,
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
    print(f"Trainable params: {model.get_num_trainable_params():,}")

    wandb.watch(model, log='all', log_freq=100)

    criterion = create_loss_function(config)
    optimizer = create_optimizer(model, config)

    use_amp = config['training'].get('mixed_precision', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision (float16) enabled")

    # OneCycleLR needs steps_per_epoch at init; inject it before create_scheduler.
    if config['scheduler']['type'].lower() == 'one_cycle':
        config['_one_cycle_steps_per_epoch'] = len(train_loader)

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

                print(f"Trainable parameters: {model.get_num_trainable_params():,}")
                print(f"{'='*60}\n")

        train_loss, avg_grad_norm, max_grad_norm = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config, scheduler, scaler
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, device, epoch, use_amp
        )

        unfreeze_epoch = config['model']['gradual_unfreeze'].get('unfreeze_at_epoch', -1)
        backbone_frozen = (
            config['model']['gradual_unfreeze']['enabled'] and epoch < unfreeze_epoch
        )

        metrics_to_log = {
            'epoch': epoch,
            'train/loss': train_loss,
            'train/grad_norm_avg': avg_grad_norm,
            'train/grad_norm_max': max_grad_norm,
            'val/loss': val_loss,
            'val/accuracy': val_metrics['accuracy'],
            'val/exact_match': val_metrics['exact_match'],
            'val/mae': val_metrics['mae'],
            'learning_rate': optimizer.param_groups[0]['lr'],
            'backbone_frozen': int(backbone_frozen),
        }

        if device.type == 'cuda':
            metrics_to_log['gpu/memory_allocated_mb'] = torch.cuda.memory_allocated(device) / 1e6
            metrics_to_log['gpu/memory_reserved_mb'] = torch.cuda.memory_reserved(device) / 1e6

        emotions = dataset_info['emotion_columns']
        class_names_per_emotion = config['data']['class_names']
        all_preds = val_metrics['all_preds']      # (N, num_emotions)
        all_targets = val_metrics['all_targets']  # (N, num_emotions)

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
            targets_i = all_targets[:, i].tolist()
            cls_names = class_names_per_emotion[i]
            metrics_to_log[f'val/confusion_matrix_{emotion}'] = wandb.plot.confusion_matrix(
                y_true=targets_i, preds=preds_i, class_names=cls_names
            )
            metrics_to_log[f'val/pred_dist_{emotion}'] = wandb.Histogram(preds_i)

        if 'f1' in val_metrics:
            metrics_to_log['val/precision'] = val_metrics['precision']
            metrics_to_log['val/recall'] = val_metrics['recall']
            metrics_to_log['val/f1'] = val_metrics['f1']
            for i, emotion in enumerate(emotions):
                metrics_to_log[f'val/f1_{emotion}'] = val_metrics['f1_per_emotion'][i]

        wandb.log(metrics_to_log)

        pred_table_freq = config.get('wandb', {}).get('prediction_table_freq', 5)
        if (epoch + 1) % pred_table_freq == 0:
            _log_prediction_table(model, val_loader, device, emotions, config)

        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}, Exact Match: {val_metrics['exact_match']:.4f}, MAE: {val_metrics['mae']:.4f}")

        if config['scheduler']['type'].lower() == 'reduce_on_plateau':
            scheduler.step(val_loss)
        elif config['scheduler']['type'].lower() != 'one_cycle':
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = run_checkpoint_dir / 'best_model.pth'
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                train_loss, val_loss, str(best_checkpoint_path),
                best_val_loss, config
            )
            print(f"Saved best model: epoch={epoch+1}, val_loss={val_loss:.4f}")

        if not config['output']['save_best_only'] and (epoch + 1) % config['output']['save_frequency'] == 0:
            checkpoint_path = run_checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
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
    print("Please use scripts/train_vit.py to run training")
    sys.exit(1)

