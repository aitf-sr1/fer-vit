# FER - Transformer

A Facial Emotion Recognition system that classifies four student engagement-related emotions — **Boredom, Engagement, Confusion, and Frustration** — using Vision Transformer-based architectures. The project provides two independent model pipelines: one that operates on raw face images, and one that operates on facial landmarks extracted from those images.

---

## What This Project Does

Each pipeline takes face data as input and outputs per-emotion classification logits with shape `(batch, 4 emotions, N classes)`. The four emotions are treated as separate multi-class classification heads on a shared backbone, allowing each emotion to be predicted independently.

---

## File Structure

```
fer-vit/
├── data/                          # Datasets (CSV labels + image folders)
│   └── landmarks/                 # Pre-extracted landmark CSVs
├── outputs/                       # Checkpoints, logs, evaluation results
├── scripts/                       # Entry-point scripts for training and evaluation
│   ├── train_vit.py
│   ├── train_landmark.py
│   ├── train_sweep.py
│   ├── evaluate_vit.py
│   ├── evaluate_landmark.py
│   ├── evaluate_latest.py
│   └── download_farl.py
├── src/
│   ├── vit/                       # Image-based pipeline
│   │   ├── config/                # YAML config files (base + variants + sweeps)
│   │   ├── data/                  # Dataset classes and transforms
│   │   ├── models/                # ViTEmotionModel
│   │   ├── training/              # Training loop, losses, optimizers, utils
│   │   └── inference/             # Evaluation and attention rollout
│   └── transformer_landmark/      # Landmark-based pipeline
│       ├── config/                # YAML config files
│       ├── data/                  # Landmark dataset and face mesh topology
│       ├── models/                # FacialLandmarkGraphTransformer, LandmarkViTModel
│       ├── training/              # Training loop, losses, optimizers, utils
│       └── inference/             # Evaluation
├── pyproject.toml
└── .env.example
```

---

## Architecture Overview

### Image Pipeline (`src/vit/`)

The `ViTEmotionModel` accepts 224x224 RGB face images. It supports four backbone variants:

- `imagenet_vit` — torchvision ViT-B/16 pretrained on ImageNet
- `farl` — face-aware ViT-B/16 loaded from a FaRL CLIP checkpoint
- `davit` — DaViT loaded via `timm`
- `efficientvit` — EfficientViT-M2 loaded via `timm` (config key: `efficientvit_variant`, default `efficientvit_m2`). A hierarchical backbone with no CLS token; visualization uses GradCAM instead of attention rollout.

The backbone's classification head is replaced with `N` independent linear heads (one per emotion). The backbone can be frozen initially and gradually unfrozen at a configured epoch.

### Landmark Pipeline (`src/transformer_landmark/`)

Two models are available:

- `FacialLandmarkGraphTransformer` — takes 478 MediaPipe (x, y) landmarks as a graph. Nodes are connected by the face mesh topology (face oval, eyes, iris). Graph Transformer layers (`TransformerConv`) process the graph, a readout pools node features to a single vector, then per-emotion linear heads produce the final logits.
- `LandmarkViTModel` — projects the flattened landmark vector back to a 224x224 synthetic image and feeds it through ViT-B/16. This is an alternative that reuses the image backbone without a graph.

### Shared Components

Both pipelines share the same loss, optimizer, scheduler, and metric infrastructure:

- **Losses:** `MultiHeadCrossEntropyLoss`, `MultiHeadFocalLoss`, ASL, BCE — all averaged across the four emotion heads
- **Optimizers:** Adam, AdamW, SGD, RMSprop
- **Schedulers:** `ReduceLROnPlateau`, `CosineAnnealingWarmRestarts`, `OneCycleLR`, `CosineAnnealingLR`, `StepLR`
- **Training utilities:** gradient accumulation, AMP (mixed precision), gradient clipping, early stopping, W&B logging

---

## Configuration

All training behaviour is controlled by YAML config files. The default configs are:

- `src/vit/config/base_config.yaml` — image pipeline
- `src/transformer_landmark/config/base_config.yaml` — landmark pipeline

The config is divided into sections:

| Section | Description |
|---|---|
| `model` | Backbone type, pretrained weights, freeze settings, gradual unfreeze |
| `data` | Paths to train/val/test CSVs and image directories |
| `augmentation` | Per-split augmentation flags and ImageNet normalization stats |
| `training` | Batch size, epochs, learning rate, weight decay, gradient clip |
| `loss` | Loss type and parameters (label smoothing, class weights, focal gamma) |
| `optimizer` | Optimizer type and hyperparameters |
| `scheduler` | Scheduler type and hyperparameters |
| `early_stopping` | Patience and minimum delta |
| `output` | Checkpoint and log output directories |
| `device` | CUDA device selection |

To use a different configuration, pass it via `--config`. Any pre-built configs (e.g. `farl_binary.yaml`, `davit_binary.yaml`, `efficientvit_first_data.yaml`) in `src/vit/config/` can be used as-is or as a starting point.

For EfficientViT-M2, set the following in the `model` section:

```yaml
model:
  backbone: efficientvit
  efficientvit_variant: efficientvit_m2  # any timm EfficientViT variant
  pretrained: true
```

---

## Quick Start

### 1. Setup

**Install dependencies** using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

**Configure environment variables** by copying the example file and filling in your W&B credentials:

```bash
cp .env.example .env
```

```
WANDB_API_KEY=your_wandb_api_key_here
WANDB_ENTITY=your_wandb_username_or_team
WANDB_PROJECT=fer-transformer
```

**Prepare data** by placing your CSV label files and image folders under `data/` to match the paths in the config:

```
data/
├── Cleaned_TrainLabels.csv
├── Cleaned_ValidationLabels.csv
├── Cleaned_TestLabels.csv
├── Train/
├── Validation/
└── Test/
```

For the landmark pipeline, pre-extracted landmark CSVs go in `data/landmarks/`.

### 2. Training

**Image pipeline (ViT):**

```bash
python scripts/train_vit.py --config src/vit/config/base_config.yaml
```

**Landmark pipeline (Graph Transformer):**

```bash
python scripts/train_landmark.py --config src/transformer_landmark/config/base_config.yaml
```

Checkpoints are saved to the directory specified in `output.checkpoint_dir` in the config.

### 3. Sweeps

A sweep runs multiple training configs sequentially in a single command — useful for ablations over loss functions, backbones, or hyperparameters. Each config is an independent W&B run with its own checkpoint directory.

**Run a sweep:**

```bash
python scripts/train_sweep.py \
  --configs src/vit/config/sweeps/davit_ce_weighted.yaml \
            src/vit/config/sweeps/davit_focal_g1.yaml \
            src/vit/config/sweeps/davit_asl.yaml \
  --tags sweep-01 loss-ablation
```

Tags passed via `--tags` are merged into every run's W&B tag list.

**Writing a sweep config:**

A sweep config is a regular training config with two extra top-level fields:

```yaml
experiment_name: "my-experiment"   # used as the W&B run name
tags:
  - davit
  - binary
  - cross-entropy
```

Set a unique `output.checkpoint_dir` per config so runs do not overwrite each other:

```yaml
output:
  checkpoint_dir: outputs/sweeps/my_experiment
  log_dir: outputs/sweeps/my_experiment/logs
```

Pre-built sweep configs for the image pipeline live in `src/vit/config/sweeps/`.

---

### 4. Evaluation

**Image pipeline:**

```bash
python scripts/evaluate_vit.py \
  --checkpoint outputs/checkpoints/best_model.pt \
  --config src/vit/config/base_config.yaml \
  --output outputs/evaluation_results.json
```

To also generate attention rollout heatmaps:

```bash
python scripts/evaluate_vit.py \
  --checkpoint outputs/checkpoints/best_model.pt \
  --attention-maps \
  --attention-samples 16
```

**Landmark pipeline:**

```bash
python scripts/evaluate_landmark.py \
  --checkpoint outputs/checkpoints_landmark/best_model.pt \
  --config src/transformer_landmark/config/base_config.yaml \
  --output outputs/evaluation_results_landmark.json
```
