#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml

from src.vit.models.vit_model import create_model
from src.vit.training.utils import load_checkpoint


class _ImageOnlyWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, aux=None)


class _MultimodalWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        return self.model(x, aux=aux)


def _resolve_config(checkpoint_path: str, config_path: str | None) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'config' in checkpoint:
        print("Using config embedded in checkpoint.")
        return checkpoint['config']
    if config_path is None:
        raise ValueError(
            "Checkpoint has no embedded config. Provide --config explicitly."
        )
    with open(config_path) as f:
        print(f"Using config from: {config_path}")
        return yaml.safe_load(f)


def _aux_dim(config: dict) -> int | None:
    aux_cfg = config.get('model', {}).get('auxiliary', {})
    if not aux_cfg.get('enabled', False):
        return None
    from src.vit.models.auxiliary_encoder import LandmarkEncoder
    aux_type = aux_cfg['type']
    embed_dim = aux_cfg.get('embed_dim', 128)
    if aux_type == 'mediapipe_landmarks':
        return LandmarkEncoder.INPUT_DIM
    if aux_type == 'action_units':
        return len(config['data'].get('au_columns', []))
    if aux_type == 'both':
        return LandmarkEncoder.INPUT_DIM + len(config['data'].get('au_columns', []))
    return None


def export(
    checkpoint_path: str,
    output_path: str,
    config_path: str | None,
    image_size: int,
    batch_size: int,
    opset: int,
    fp16: bool,
) -> None:
    config = _resolve_config(checkpoint_path, config_path)
    config['model']['freeze_backbone'] = False

    model = create_model(config)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    if fp16:
        model = model.half()

    dtype = torch.float16 if fp16 else torch.float32
    dummy_image = torch.zeros(batch_size, 3, image_size, image_size, dtype=dtype)
    aux_dim = _aux_dim(config)
    is_multimodal = aux_dim is not None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if is_multimodal:
        dummy_aux = torch.zeros(batch_size, aux_dim, dtype=dtype)
        export_model = _MultimodalWrapper(model)
        input_names = ['image', 'aux']
        dynamic_axes = {
            'image': {0: 'batch'},
            'aux': {0: 'batch'},
            'logits': {0: 'batch'},
        }
        dummy_inputs = (dummy_image, dummy_aux)
    else:
        export_model = _ImageOnlyWrapper(model)
        input_names = ['image']
        dynamic_axes = {
            'image': {0: 'batch'},
            'logits': {0: 'batch'},
        }
        dummy_inputs = (dummy_image,)

    print(f"Exporting {'multimodal' if is_multimodal else 'image-only'} model to ONNX...")
    print(f"  Input image shape : {list(dummy_image.shape)}")
    if is_multimodal:
        print(f"  Input aux shape   : {list(dummy_aux.shape)}")
    print(f"  Opset             : {opset}")
    print(f"  Precision         : {'fp16' if fp16 else 'fp32'}")

    torch.onnx.export(
        export_model,
        dummy_inputs,
        str(output_path),
        input_names=input_names,
        output_names=['logits'],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    print(f"Saved: {output_path}")
    print(f"Output shape: (batch, {config['model']['num_emotions']}, {config['model'].get('num_classes', 4)})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export a trained ViT model to ONNX')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to .pt checkpoint file')
    parser.add_argument('--output', default='outputs/model.onnx',
                        help='Output .onnx file path (default: outputs/model.onnx)')
    parser.add_argument('--config', default=None,
                        help='Path to YAML config (optional if config is embedded in checkpoint)')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size (default: 224)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for dummy input; use 1 for dynamic batching (default: 1)')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version (default: 17)')
    parser.add_argument('--fp16', action='store_true',
                        help='Export in FP16 precision (requires FP16-capable runtime)')
    args = parser.parse_args()

    export(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        config_path=args.config,
        image_size=args.image_size,
        batch_size=args.batch_size,
        opset=args.opset,
        fp16=args.fp16,
    )
