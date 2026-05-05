#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from src.transformer_landmark.inference.evaluate import evaluate


def find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    latest_dir = checkpoint_dir / 'latest'
    
    if not latest_dir.exists():
        raise FileNotFoundError(f"No training runs found in {checkpoint_dir}/")
    
    best_models = list(latest_dir.glob('best_model_*.pth'))
    
    if not best_models:
        raise FileNotFoundError(f"No best model checkpoint found in {latest_dir}/")
    
    best_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return best_models[0]


def main():
    parser = argparse.ArgumentParser(description='Evaluate the latest trained model')
    parser.add_argument('--checkpoint-dir', type=str, 
                        default='outputs/checkpoints_landmark',
                        help='Base checkpoint directory')
    parser.add_argument('--config', type=str, 
                        default='src/transformer_landmark/config/base_config.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, 
                        default='outputs/evaluation_results_landmark.json',
                        help='Path to save evaluation results')
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    
    try:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train a model first using: python scripts/train_landmark.py")
        sys.exit(1)
    
    print("=" * 60)
    print("Evaluating latest model:")
    print(f"  Checkpoint: {checkpoint_path}")
    print("=" * 60)
    print()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    evaluate(str(checkpoint_path), config, args.output)


if __name__ == "__main__":
    main()
