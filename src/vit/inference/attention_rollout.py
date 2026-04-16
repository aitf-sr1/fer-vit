"""
Attention rollout for vit_b_16 (torchvision).

Rolls up attention weights across all 12 encoder layers following:
  Abnar & Zuidema (2020) "Quantifying Attention Flow in Transformers"

The result is a 14x14 spatial relevance map for the [CLS] token,
which can be upsampled and overlaid on the original image.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class AttentionRollout:
    def __init__(self, model: nn.Module, discard_ratio: float = 0.9):
        """
        Args:
            model: ViTEmotionModel instance.
            discard_ratio: fraction of lowest attention weights to zero out
                           before rollout, reducing noise.
        """
        self.model = model
        self.discard_ratio = discard_ratio
        self._attention_weights: List[torch.Tensor] = []
        self._hooks: list = []

    def _hook_fn(self, module, input, output):
        # torchvision MultiheadAttention returns (attn_output, attn_weights)
        # attn_weights shape: (batch, num_heads, seq_len, seq_len)
        if isinstance(output, tuple) and len(output) == 2:
            self._attention_weights.append(output[1].detach().cpu())

    def _register_hooks(self):
        for layer in self.model.vit.encoder.layers:
            hook = layer.self_attention.register_forward_hook(self._hook_fn)
            self._hooks.append(hook)

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _rollout(self) -> torch.Tensor:
        # Stack: (num_layers, batch, num_heads, seq, seq)
        attentions = torch.stack(self._attention_weights, dim=0)
        # Average over heads: (num_layers, batch, seq, seq)
        attentions = attentions.mean(dim=2)

        batch_size, seq_len = attentions.shape[1], attentions.shape[2]

        # Add identity (residual connection) and re-normalize rows
        eye = torch.eye(seq_len).unsqueeze(0).unsqueeze(0)
        attentions = (attentions + eye) / 2
        attentions = attentions / attentions.sum(dim=-1, keepdim=True)

        # Discard lowest attention values to reduce noise
        flat = attentions.view(-1, seq_len)
        threshold = flat.quantile(self.discard_ratio, dim=-1, keepdim=True)
        attentions = torch.where(attentions >= threshold.view(*attentions.shape[:-1], 1),
                                 attentions,
                                 torch.zeros_like(attentions))
        attentions = attentions / (attentions.sum(dim=-1, keepdim=True) + 1e-8)

        # Multiply across layers: result shape (batch, seq, seq)
        result = attentions[0]
        for i in range(1, attentions.shape[0]):
            result = torch.bmm(attentions[i], result)

        # CLS token's attention to all patch tokens: (batch, seq)
        cls_attn = result[:, 0, 1:]  # exclude CLS-to-CLS
        return cls_attn

    @torch.no_grad()
    def __call__(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Args:
            image_tensor: (1, C, H, W) on the same device as the model.

        Returns:
            rollout map as numpy array of shape (14, 14), values in [0, 1].
        """
        self._attention_weights.clear()
        self._register_hooks()

        # torchvision MultiheadAttention does not return attn_weights by default;
        # we need need_weights=True (the default), but batch_first matters.
        # Patch: temporarily enable need_weights on all attention layers.
        original_flags = {}
        for name, module in self.model.vit.encoder.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                original_flags[name] = module.training
        # Forward pass
        try:
            _ = self.model(image_tensor)
        finally:
            self._remove_hooks()

        if not self._attention_weights:
            raise RuntimeError(
                "No attention weights captured. "
                "The torchvision MHA may not be returning weights — "
                "check that need_weights=True (default)."
            )

        cls_attn = self._rollout()  # (1, 196)
        grid_size = int(cls_attn.shape[-1] ** 0.5)  # 14
        attn_map = cls_attn[0].reshape(grid_size, grid_size).numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        return attn_map


def overlay_attention_on_image(
    original_image: Image.Image,
    attn_map: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Upsample attn_map to image size and overlay as a heatmap.

    Returns an RGB numpy array (uint8).
    """
    img_np = np.array(original_image.convert("RGB"))
    h, w = img_np.shape[:2]

    attn_resized = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(
        (attn_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)
    return overlay


def save_attention_maps(
    model: nn.Module,
    dataset,
    device: torch.device,
    emotion_columns: List[str],
    output_dir: str,
    num_samples: int = 16,
    discard_ratio: float = 0.9,
) -> None:
    """
    Generates and saves attention rollout heatmaps for `num_samples` images.

    Args:
        model: trained ViTEmotionModel.
        dataset: dataset instance (EmotionDataset or BinaryEmotionDataset).
        device: torch device.
        emotion_columns: list of emotion label names.
        output_dir: directory to save images.
        num_samples: how many samples to visualize.
        discard_ratio: fraction of lowest attention weights to zero out.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rollout = AttentionRollout(model, discard_ratio=discard_ratio)
    model.eval()

    indices = list(range(min(num_samples, len(dataset))))

    for idx in indices:
        image_tensor, labels = dataset[idx]
        image_tensor = image_tensor.unsqueeze(0).to(device)

        attn_map = rollout(image_tensor)

        # Reconstruct original PIL image (undo normalization for display)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        display_tensor = image_tensor.squeeze(0).cpu() * std + mean
        display_tensor = display_tensor.clamp(0, 1)
        original_pil = Image.fromarray((display_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))

        overlay = overlay_attention_on_image(original_pil, attn_map)

        # Build label string for title
        preds = model(image_tensor).argmax(dim=2).squeeze(0).cpu().tolist()
        true_labels = labels.tolist()
        label_str = "  ".join(
            f"{e}: pred={p} true={t}"
            for e, p, t in zip(emotion_columns, preds, true_labels)
        )

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(original_pil)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(attn_map, cmap="jet")
        axes[1].set_title("Attention Map (14x14)")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        fig.suptitle(label_str, fontsize=9)
        plt.tight_layout()
        save_file = out_path / f"attention_sample_{idx:04d}.png"
        plt.savefig(save_file, dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {len(indices)} attention maps to: {out_path}")
