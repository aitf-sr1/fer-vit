import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from ..data.dataloader import create_dataloaders, get_dataset_info, _dataset_class
from ..models.vit_model import create_model
from ..training.utils import load_checkpoint, calculate_metrics
from .attention_rollout import save_attention_maps


def _binary_confusion_matrix(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(targets, preds):
        cm[int(t)][int(p)] += 1
    return cm


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
        'confusion_matrices': {},
    }

    if 'f1' in metrics:
        results['overall_metrics']['precision'] = metrics['precision']
        results['overall_metrics']['recall'] = metrics['recall']
        results['overall_metrics']['f1'] = metrics['f1']

    for i, emotion in enumerate(emotion_columns):
        em: Dict[str, float] = {
            'accuracy': float(metrics['accuracy_per_emotion'][i]),
            'mae': float(metrics['mae_per_emotion'][i]),
        }
        if 'f1_per_emotion' in metrics:
            em['precision'] = float(metrics['precision_per_emotion'][i])
            em['recall'] = float(metrics['recall_per_emotion'][i])
            em['f1'] = float(metrics['f1_per_emotion'][i])
        results['per_emotion_metrics'][emotion] = em

        cm = _binary_confusion_matrix(preds_np[:, i], targets_np[:, i])
        results['confusion_matrices'][emotion] = cm.tolist()

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

    if 'confusion_matrices' in results:
        print("\nConfusion Matrices (rows=actual, cols=predicted):")
        for emotion, cm in results['confusion_matrices'].items():
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            total = tn + fp + fn + tp
            print(f"\n  {emotion}:")
            print(f"              Pred 0   Pred 1")
            print(f"    Actual 0   {tn:5d}    {fp:5d}")
            print(f"    Actual 1   {fn:5d}    {tp:5d}")
            print(f"    TPR(recall): {tp/(tp+fn):.3f}  FPR: {fp/(fp+tn):.3f}  Total: {total}")

    print("\n" + "="*60)


def plot_confusion_matrices(results: Dict[str, Any], output_path: str) -> None:
    cms = results.get('confusion_matrices', {})
    if not cms:
        return

    emotions = list(cms.keys())
    n = len(emotions)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, emotion in zip(axes, emotions):
        cm = np.array(cms[emotion])
        total = cm.sum()
        norm_cm = cm.astype(float) / (total if total > 0 else 1)

        im = ax.imshow(norm_cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        ax.set_title(emotion, fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['0', '1'])
        ax.set_yticklabels(['0', '1'])

        for row in range(2):
            for col in range(2):
                count = cm[row, col]
                pct = norm_cm[row, col]
                color = 'white' if pct > 0.5 else 'black'
                ax.text(col, row, f"{count}\n({pct:.1%})", ha='center', va='center',
                        fontsize=10, color=color)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Confusion Matrices per Emotion', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Confusion matrices saved to: {output_path}")


def save_results(results: Dict[str, Any], output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results_to_save = {
        'overall_metrics': results['overall_metrics'],
        'per_emotion_metrics': results['per_emotion_metrics'],
        'confusion_matrices': results.get('confusion_matrices', {}),
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

    cm_plot_path = output_file.parent / f"{output_file.stem}_confusion_matrices.png"
    plot_confusion_matrices(results, str(cm_plot_path))


def evaluate(
    checkpoint_path: str,
    config: Dict[str, Any],
    output_path: str,
    attention_maps: bool = False,
    attention_samples: int = 16,
    attention_output_dir: Optional[str] = None,
    attention_discard_ratio: float = 0.7,
) -> None:
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

    if attention_maps:
        attn_dir = attention_output_dir or str(Path(output_path).parent / "attention_maps")
        from ..data.transforms import get_test_transforms
        transform = get_test_transforms(config)
        dataset_cls = _dataset_class(config)
        extra = _dataset_kwargs(config)
        data_cfg = config['data']
        test_dataset = dataset_cls(
            csv_file=data_cfg['test_csv'],
            img_dir=data_cfg['test_img_dir'],
            transform=transform,
            **extra,
        )
        print(f"\nGenerating {attention_samples} attention maps...")
        save_attention_maps(
            model=model,
            dataset=test_dataset,
            device=device,
            emotion_columns=dataset_info['emotion_columns'],
            output_dir=attn_dir,
            num_samples=attention_samples,
            discard_ratio=attention_discard_ratio,
        )

if __name__ == '__main__':
    import sys
    print("Please use scripts/evaluate_vit.py to run evaluation")
    sys.exit(1)
