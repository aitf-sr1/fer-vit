from pathlib import Path
from typing import Dict, Any, List
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from ..data.dataloader import create_dataloaders, get_dataset_info
from ..models import create_model
from ..training.utils import load_checkpoint, calculate_metrics


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    emotion_columns: list[str],
    mode: str = "classification"
) -> Dict[str, Any]:
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating")
        for landmarks, labels in progress_bar:
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            
            outputs = model(landmarks)
            all_predictions.append(outputs.cpu())
            all_targets.append(labels.cpu())
    
    all_predictions_tensor = torch.cat(all_predictions, dim=0)
    all_targets_tensor = torch.cat(all_targets, dim=0)
    
    metrics = calculate_metrics(all_predictions_tensor, all_targets_tensor, mode=mode)
    
    if mode == "classification":
        preds_np = all_predictions_tensor.argmax(dim=2).numpy()
        targets_np = all_targets_tensor.numpy()
        
        results = {
            'overall_metrics': {
                'accuracy': metrics['accuracy'],
                'exact_match': metrics['exact_match'],
                'mae': metrics['mae']
            },
            'per_emotion_metrics': {}
        }
        
        for i, emotion in enumerate(emotion_columns):
            acc_list = metrics['accuracy_per_emotion']
            mae_list = metrics['mae_per_emotion']
            results['per_emotion_metrics'][emotion] = {
                'accuracy': float(acc_list[i]) if isinstance(acc_list, list) else float(acc_list),
                'mae': float(mae_list[i]) if isinstance(mae_list, list) else float(mae_list)
            }
        
        results['predictions'] = preds_np.tolist()
        results['targets'] = targets_np.tolist()
    else:
        predictions_np = all_predictions_tensor.numpy()
        targets_np = all_targets_tensor.numpy()
        
        results = {
            'overall_metrics': {
                'mse': metrics['mse'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse']
            },
            'per_emotion_metrics': {}
        }
        
        for i, emotion in enumerate(emotion_columns):
            mse_per_emotion: List[float] = metrics['mse_per_emotion']
            mae_per_emotion: List[float] = metrics['mae_per_emotion']
            corr_per_emotion: List[float] = metrics['correlation_per_emotion']
            results['per_emotion_metrics'][emotion] = {
                'mse': float(mse_per_emotion[i]),
                'mae': float(mae_per_emotion[i]),
                'rmse': float(np.sqrt(mse_per_emotion[i])),
                'correlation': float(corr_per_emotion[i])
            }
        
        results['predictions'] = predictions_np.tolist()
        results['targets'] = targets_np.tolist()
    
    return results


def print_results(results: Dict[str, Any], mode: str = "classification") -> None:
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\nOverall Metrics:")
    if mode == "classification":
        print(f"  Accuracy:    {results['overall_metrics']['accuracy']:.4f}")
        print(f"  Exact Match: {results['overall_metrics']['exact_match']:.4f}")
        print(f"  MAE:         {results['overall_metrics']['mae']:.4f}")
    else:
        print(f"  MSE:  {results['overall_metrics']['mse']:.4f}")
        print(f"  MAE:  {results['overall_metrics']['mae']:.4f}")
        print(f"  RMSE: {results['overall_metrics']['rmse']:.4f}")
    
    print("\nPer-Emotion Metrics:")
    for emotion, emotion_metrics in results['per_emotion_metrics'].items():
        print(f"\n{emotion}:")
        if mode == "classification":
            print(f"  Accuracy: {emotion_metrics['accuracy']:.4f}")
            print(f"  MAE:      {emotion_metrics['mae']:.4f}")
        else:
            print(f"  MSE:         {emotion_metrics['mse']:.4f}")
            print(f"  MAE:         {emotion_metrics['mae']:.4f}")
            print(f"  RMSE:        {emotion_metrics['rmse']:.4f}")
            print(f"  Correlation: {emotion_metrics['correlation']:.4f}")
    
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
    
    _, _, test_loader = create_dataloaders(config)
    dataset_info = get_dataset_info(config)
    print(f"Test set size: {dataset_info['test_size']}")
    
    model = create_model(config)
    load_checkpoint(checkpoint_path, model)
    model = model.to(device)
    print(f"Model loaded from: {checkpoint_path}")
    
    mode = config.get('model', {}).get('mode', 'classification')
    results = evaluate_model(model, test_loader, device, dataset_info['emotion_columns'], mode=mode)
    
    print_results(results, mode=mode)
    
    if output_path:
        save_results(results, output_path)

if __name__ == '__main__':
    import sys
    print("Please use scripts/evaluate_landmark.py to run evaluation")
    sys.exit(1)
