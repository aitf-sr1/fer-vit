# ViT Config Structure

This directory organizes training configurations for the image-based facial emotion recognition pipeline.

## Directory Organization

```
config/
├── base/                      # Base/reference configurations
│   ├── base_config.yaml       # Default ViT-B/16 config
│   ├── dinov3_base.yaml       # DINOv3 ViT-S/16 base config
│   └── dinov3_vitb_base.yaml  # DINOv3 ViT-B/16 base config
│
├── backbones/                 # Backbone-specific training configs
│   ├── davit/
│   │   ├── binary.yaml        # DaViT on binary dataset
│   │   └── first_data.yaml    # DaViT on first_data dataset
│   ├── farl/
│   │   └── binary.yaml        # FaRL on binary dataset
│   ├── efficientvit/
│   │   └── first_data.yaml    # EfficientViT on first_data dataset
│   └── dinov3/                # (Configs managed via sweeps)
│
├── datasets/                  # Dataset-specific configs
│   ├── binary_dataset.yaml    # Binary classification (4→2 classes)
│   └── vit-1.yaml             # Alternative config
│
└── sweeps/                    # Hyperparameter sweep configs (loss ablations)
    ├── davit/                 # Loss ablations for DaViT backbone
    │   ├── ce_weighted.yaml
    │   ├── focal_g1.yaml
    │   ├── focal_g2.yaml
    │   ├── asl.yaml
    │   └── bce.yaml
    │
    ├── farl/                  # Loss ablations for FaRL backbone
    │   ├── ce_weighted.yaml
    │   ├── focal_g1.yaml
    │   ├── focal_g2.yaml
    │   ├── asl.yaml
    │   └── bce.yaml
    │
    ├── dinov3/                # Loss ablations for DINOv3
    │   ├── vit_s/             # DINOv3 ViT-S/16 sweeps
    │   │   ├── ce_weighted.yaml
    │   │   ├── ce_weighted_exactmatch.yaml
    │   │   ├── focal_g1.yaml
    │   │   ├── focal_g2.yaml
    │   │   ├── asl.yaml
    │   │   └── bce.yaml
    │   └── vit_b/             # DINOv3 ViT-B/16 sweeps
    │       ├── ce_weighted.yaml
    │       ├── ce_weighted_exactmatch.yaml
    │       ├── focal_g1.yaml
    │       ├── focal_g2.yaml
    │       ├── asl.yaml
    │       └── bce.yaml
    │
    └── efficientvit/          # Loss ablations for EfficientViT
        └── first_data/        # EfficientViT on first_data dataset
            ├── ce_weighted.yaml
            ├── ce_weighted_adamw.yaml
            ├── ce_weighted_adamw_exact_match.yaml
            ├── focal_g1.yaml
            ├── focal_g2.yaml
            ├── asl.yaml
            └── bce.yaml
```

## Usage

### Training with Base Configs

```bash
# Default ViT-B/16 (ImageNet)
python scripts/train_vit.py --config src/vit/config/base/base_config.yaml

# DINOv3 ViT-S/16
python scripts/train_vit.py --config src/vit/config/base/dinov3_base.yaml

# DINOv3 ViT-B/16
python scripts/train_vit.py --config src/vit/config/base/dinov3_vitb_base.yaml
```

### Training with Backbone Configs

```bash
# DaViT on binary dataset
python scripts/train_vit.py --config src/vit/config/backbones/davit/binary.yaml

# FaRL on binary dataset
python scripts/train_vit.py --config src/vit/config/backbones/farl/binary.yaml

# EfficientViT on first_data dataset
python scripts/train_vit.py --config src/vit/config/backbones/efficientvit/first_data.yaml
```

### Running Loss Ablation Sweeps

```bash
# DaViT loss ablation sweep
python scripts/train_sweep.py \
  --configs src/vit/config/sweeps/davit/ce_weighted.yaml \
            src/vit/config/sweeps/davit/focal_g1.yaml \
            src/vit/config/sweeps/davit/focal_g2.yaml \
            src/vit/config/sweeps/davit/asl.yaml \
            src/vit/config/sweeps/davit/bce.yaml \
  --tags davit loss-ablation

# DINOv3 ViT-S/16 loss ablation sweep
python scripts/train_sweep.py \
  --configs src/vit/config/sweeps/dinov3/vit_s/ce_weighted.yaml \
            src/vit/config/sweeps/dinov3/vit_s/focal_g1.yaml \
            src/vit/config/sweeps/dinov3/vit_s/focal_g2.yaml \
            src/vit/config/sweeps/dinov3/vit_s/asl.yaml \
            src/vit/config/sweeps/dinov3/vit_s/bce.yaml \
  --tags dinov3-vit-s loss-ablation

# DINOv3 ViT-B/16 loss ablation sweep
python scripts/train_sweep.py \
  --configs src/vit/config/sweeps/dinov3/vit_b/ce_weighted.yaml \
            src/vit/config/sweeps/dinov3/vit_b/focal_g1.yaml \
            src/vit/config/sweeps/dinov3/vit_b/focal_g2.yaml \
            src/vit/config/sweeps/dinov3/vit_b/asl.yaml \
            src/vit/config/sweeps/dinov3/vit_b/bce.yaml \
  --tags dinov3-vit-b loss-ablation
```

## Output Directory Structure

All configs follow a standardized output path pattern:

```
outputs/
├── checkpoints/
│   ├── davit/
│   │   ├── binary/
│   │   ├── first_data/
│   │   ├── ce_weighted/
│   │   ├── focal_g1/
│   │   ├── focal_g2/
│   │   ├── asl/
│   │   └── bce/
│   ├── farl/
│   │   ├── binary/
│   │   ├── ce_weighted/
│   │   └── ...
│   ├── dinov3_vit_s/
│   │   ├── ce_weighted/
│   │   ├── ce_weighted_exactmatch/
│   │   ├── focal_g1/
│   │   └── ...
│   ├── dinov3_vit_b/
│   │   └── ...
│   └── efficientvit_first_data/
│       ├── ce_weighted/
│       ├── focal_g1/
│       └── ...
│
└── logs/
    ├── davit/
    ├── farl/
    ├── dinov3_vit_s/
    ├── dinov3_vit_b/
    └── efficientvit_first_data/
        └── (same structure as checkpoints)
```

## Config Naming Conventions

**Base Configs:**
- Use descriptive names: `base_config.yaml`, `dinov3_base.yaml`, `dinov3_vitb_base.yaml`

**Backbone Configs:**
- Format: `{backbone}/{dataset}.yaml`
- Examples: `davit/binary.yaml`, `farl/binary.yaml`

**Sweep Configs:**
- Format: `sweeps/{backbone}/{loss_function}.yaml`
- Loss functions: `ce_weighted`, `focal_g1`, `focal_g2`, `asl`, `bce`
- For DINOv3 variants: `sweeps/dinov3/{variant}/{loss_function}.yaml`

**Output Directories:**
- Format: `outputs/{checkpoints|logs}/{backbone_variant}/{config_type|loss_name}/`
- Backbone variants: `davit`, `farl`, `dinov3_vit_s`, `dinov3_vit_b`, `efficientvit_first_data`
- Types: `binary`, `first_data`, or loss function names (for sweeps)

## Adding New Configs

When adding a new config:

1. **Determine the type:** Is it a base reference, backbone experiment, or loss ablation?
2. **Choose the location:** Place it in the appropriate directory (`base/`, `backbones/`, or `sweeps/`)
3. **Update output paths:** Ensure `output.checkpoint_dir` and `output.log_dir` follow the standardized pattern
4. **Use clear names:** For sweeps, the filename should clearly indicate the loss function and backbone

Example output configuration:

```yaml
output:
  checkpoint_dir: outputs/checkpoints/{backbone_variant}/{config_type}
  log_dir: outputs/logs/{backbone_variant}/{config_type}
  save_best_only: true
  save_frequency: 5
```

