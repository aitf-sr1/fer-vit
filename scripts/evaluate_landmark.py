#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml  # noqa: E402
from src.transformer_landmark.inference.evaluate import evaluate  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Landmark-Guided ViT model on test set')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='src/transformer_landmark/config/base_config.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default='outputs/evaluation_results_landmark.json',
                        help='Path to save evaluation results')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loading config from: {args.config}")
    evaluate(args.checkpoint, config, args.output)
