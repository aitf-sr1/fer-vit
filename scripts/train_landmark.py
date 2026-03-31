#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml  # noqa: E402
from src.transformer_landmark.training.train import train  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Landmark-Guided ViT for emotion detection')
    parser.add_argument('--config', type=str, default='src/transformer_landmark/config/base_config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loading config from: {args.config}")
    train(config)
