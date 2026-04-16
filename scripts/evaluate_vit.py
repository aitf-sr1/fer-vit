#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml  # noqa: E402
from src.vit.inference.evaluate import evaluate  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate ViT model on test set')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='src/vit/config/vit-1.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default='outputs/evaluation_results.json',
                        help='Path to save evaluation results')
    parser.add_argument('--attention-maps', action='store_true',
                        help='Generate attention rollout heatmaps after evaluation')
    parser.add_argument('--attention-samples', type=int, default=16,
                        help='Number of test images to visualize (default: 16)')
    parser.add_argument('--attention-output-dir', type=str, default=None,
                        help='Directory to save attention maps (default: outputs/attention_maps/)')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loading config from: {args.config}")
    evaluate(
        args.checkpoint,
        config,
        args.output,
        attention_maps=args.attention_maps,
        attention_samples=args.attention_samples,
        attention_output_dir=args.attention_output_dir,
    )
