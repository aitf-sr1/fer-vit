# Landmark Pipeline Config Structure

This directory contains training configurations for the landmark-based facial emotion recognition pipeline that uses facial landmarks extracted from face images.

## Directory Organization

```
config/
└── base_config.yaml           # Reference configuration for the landmark pipeline
```

## Configuration Sections

The landmark pipeline config is divided into the following sections:

| Section | Description |
|---|---|
| `model` | Landmark model type (graph transformer or landmark ViT), architecture parameters |
| `data` | Paths to train/val/test landmark CSVs and dataset configuration |
| `augmentation` | Per-split augmentation flags and normalization settings |
| `training` | Batch size, epochs, learning rate, weight decay, gradient clipping |
| `loss` | Loss type and parameters (label smoothing, class weights, focal gamma) |
| `optimizer` | Optimizer type and hyperparameters |
| `scheduler` | Learning rate scheduler type and hyperparameters |
| `early_stopping` | Patience and minimum delta for early stopping |
| `output` | Checkpoint and log output directories |
| `device` | CUDA device selection |

## Usage

### Training with Base Config

```bash
python scripts/train_landmark.py --config src/transformer_landmark/config/base_config.yaml
```

### Output Directory Structure

Checkpoints and logs are saved to:

```
outputs/
├── checkpoints_landmark/
│   └── (model checkpoints)
└── logs_landmark/
    └── (training logs and metrics)
```

## Available Models

The landmark pipeline supports two model architectures:

### FacialLandmarkGraphTransformer

A Graph Transformer that processes 478 MediaPipe landmarks as a graph:

```yaml
model:
  type: graph_transformer
  d_model: 128
  num_heads: 4
  num_layers: 4
  num_emotions: 4
  num_classes: 4
  readout: mean  # or sum, max
```

- Takes (x, y) coordinates of 478 facial landmarks
- Constructs a graph using face mesh topology (face oval, eyes, iris)
- Processes graph with Graph Transformer layers
- Pools node features to a single vector
- Outputs per-emotion classification logits

### LandmarkViTModel

A ViT-based model that converts landmarks to a synthetic image:

```yaml
model:
  type: landmark_vit
  backbone: imagenet_vit
  freeze_backbone: true
  num_emotions: 4
  num_classes: 4
```

- Projects 478 flattened landmarks to a 224×224 synthetic image
- Feeds the image through ViT-B/16
- Outputs per-emotion classification logits
- Reuses the image-based backbone architecture

## Dataset Format

Landmark data should be organized as:

```
data/
├── landmarks/
│   ├── Cleaned_TrainLabels_landmarks.csv
│   ├── Cleaned_ValidationLabels_landmarks.csv
│   └── Cleaned_TestLabels_landmarks.csv
```

Each CSV should contain:
- Landmark data (478 × 2 coordinates: x1, y1, x2, y2, ..., x478, y478)
- Emotion labels for each sample

## Config Examples

### Graph Transformer Example

```yaml
model:
  type: graph_transformer
  d_model: 128
  num_heads: 4
  num_layers: 4
  num_emotions: 4
  num_classes: 4
  readout: mean
  dropout: 0.1

data:
  train_csv: data/landmarks/Cleaned_TrainLabels_landmarks.csv
  val_csv: data/landmarks/Cleaned_ValidationLabels_landmarks.csv
  test_csv: data/landmarks/Cleaned_TestLabels_landmarks.csv

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001

loss:
  type: cross_entropy
  label_smoothing: 0.1

optimizer:
  type: adam
  betas: [0.9, 0.999]

scheduler:
  type: reduce_on_plateau
  mode: min
  factor: 0.5
  patience: 5

output:
  checkpoint_dir: outputs/checkpoints_landmark
  log_dir: outputs/logs_landmark
```

### Landmark ViT Example

```yaml
model:
  type: landmark_vit
  backbone: imagenet_vit
  pretrained: true
  freeze_backbone: true
  num_emotions: 4
  num_classes: 4

data:
  train_csv: data/landmarks/Cleaned_TrainLabels_landmarks.csv
  val_csv: data/landmarks/Cleaned_ValidationLabels_landmarks.csv
  test_csv: data/landmarks/Cleaned_TestLabels_landmarks.csv

training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.0001

loss:
  type: cross_entropy

optimizer:
  type: adamw

scheduler:
  type: reduce_on_plateau

output:
  checkpoint_dir: outputs/checkpoints_landmark
  log_dir: outputs/logs_landmark
```

## Adding New Configs

When creating a new landmark config:

1. Start from `base_config.yaml` as a template
2. Update the model section with your chosen architecture
3. Point to your landmark CSV files in the data section
4. Adjust hyperparameters as needed
5. Keep the output paths consistent: `outputs/checkpoints_landmark/` and `outputs/logs_landmark/`

