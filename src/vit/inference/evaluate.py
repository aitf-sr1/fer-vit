import argparse
from pathlib import Path
from typing import Dict, Any
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from ..data.dataloader import create_dataloaders, get_dataset_info
from ..models.vit_model import create_model
from ..training.utils import load_checkpoint, calculate_metrics


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    emotion_columns: list[str],
) -> Dict[str, Any]:
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating")
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            all_predictions.append(outputs.cpu())
            all_targets.append(labels.cpu())

    all_predictions_tensor = torch.cat(all_predictions, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)

    metrics = calculate_metrics(all_predictions_tensor, all_targets_tensor)

    preds_np = all_predictions_tensor.argmax(dim=2).numpy()
    targets_np = all_targets_tensor.numpy()

    results: Dict[str, Any] = {
        'overall_metrics': {
            'accuracy': metrics['accuracy'],
            'exact_match': metrics['exact_match'],
            'mae': metrics['mae'],
        },
        'per_emotion_metrics': {},
    }
    for i, emotion in enumerate(emotion_columns):
        results['per_emotion_metrics'][emotion] = {
            'accuracy': float(metrics['accuracy_per_emotion'][i]),
            'mae': float(metrics['mae_per_emotion'][i]),
        }

    results['predictions'] = preds_np.tolist()
    results['targets'] = targets_np.tolist()

    return results


def print_results(results: Dict[str, Any]) -> None:
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\nOverall Metrics:")
    overall = results['overall_metrics']
    for key, value in overall.items():
        print(f"  {key.upper()}: {value:.4f}")
    
    print("\nPer-Emotion Metrics:")
    for emotion, metrics in results['per_emotion_metrics'].items():
        print(f"\n{emotion}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    
    print("\n" + "="*60)


def save_results(results: Dict[str, Any], output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    results_to_save = {
        'overall_metrics': results['overall_metrics'],
        'per_emotion_metrics': results['per_emotion_metrics']
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    predictions_file = output_file.parent / f"{output_file.stem}_predictions.json"
    predictions_data = {
        'predictions': results['predictions'],
        'targets': results['targets']
    }
    
    with open(predictions_file, 'w') as f:
        json.dump(predictions_data, f, indent=2)
    
    print(f"Predictions saved to: {predictions_file}")


def evaluate(checkpoint_path: str, config: Dict[str, Any], output_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config_from_checkpoint = checkpoint.get('config')
    
    if config_from_checkpoint is not None:
        config = config_from_checkpoint
        print("Using config from checkpoint")
    
    device_config = config['device']
    if device_config['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_config['cuda_device']}")
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = create_dataloaders(config)
    dataset_info = get_dataset_info(config)
    print(f"Test set size: {dataset_info['test_size']}")
    
    model = create_model(config)
    load_checkpoint(checkpoint_path, model)
    model = model.to(device)
    print(f"Model loaded from: {checkpoint_path}")

    results = evaluate_model(model, test_loader, device, dataset_info['emotion_columns'])
    
    print_results(results)
    
    if output_path:
        save_results(results, output_path)

if __name__ == '__main__':
    import sys
    print("Please use scripts/evaluate_vit.py to run evaluation")
    sys.exit(1)
