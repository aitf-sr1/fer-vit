"""
Attention rollout for ViTEmotionModel (torchvision or FaRL/CLIP backbone).
GradCAM for DaViT backbone (hierarchical, no CLS token).

Attention rollout rolls up attention weights across all 12 encoder layers following:
  Abnar & Zuidema (2020) "Quantifying Attention Flow in Transformers"

GradCAM hooks the last DaViT stage output (C x H x W feature map before global pool)
and weights it by the gradients from the target emotion head logit.
"""

from pathlib import Path
from typing import List, Optional
import types

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class AttentionRollout:
    def __init__(self, model: nn.Module, discard_ratio: float = 0.7):
        """
        Args:
            model: ViTEmotionModel instance.
            discard_ratio: fraction of lowest attention weights to zero out
                           before rollout, reducing noise.
        """
        self.model = model
        self.discard_ratio = discard_ratio
        self._attention_weights: List[torch.Tensor] = []
        self._original_forwards = {}

    # ------------------------------------------------------------------
    # torchvision ViT patching
    # ------------------------------------------------------------------

    def _patch_encoder_blocks(self):
        """
        Patch torchvision EncoderBlock.forward to capture attention weights.
        torchvision hardcodes need_weights=False, so hooks alone cannot work.
        """
        storage = self._attention_weights

        def _patched_forward(self_block, input: torch.Tensor) -> torch.Tensor:
            x = self_block.ln_1(input)
            attn_out, attn_weights = self_block.self_attention(
                x, x, x, need_weights=True, average_attn_weights=True
            )
            if attn_weights is not None:
                storage.append(attn_weights.detach().cpu())
            x = self_block.dropout(attn_out)
            x = x + input
            y = self_block.ln_2(x)
            y = self_block.mlp(y)
            return x + y

        for i, layer in enumerate(self.model.vit.encoder.layers):
            self._original_forwards[i] = layer.forward
            layer.forward = types.MethodType(_patched_forward, layer)

    def _restore_encoder_blocks(self):
        for i, layer in enumerate(self.model.vit.encoder.layers):
            layer.forward = self._original_forwards[i]
        self._original_forwards.clear()

    # ------------------------------------------------------------------
    # CLIP / FaRL ViT patching
    # ------------------------------------------------------------------

    def _patch_encoder_blocks_farl(self):
        """
        Patch open_clip ResidualAttentionBlock.forward to capture attention
        weights. open_clip also hardcodes need_weights=False.
        """
        storage = self._attention_weights

        def _patched_forward_clip(self_block, q_x, k_x=None, v_x=None, attn_mask=None):
            k_x = k_x if k_x is not None else q_x
            v_x = v_x if v_x is not None else q_x
            ln_q = self_block.ln_1(q_x)
            attn_mask_cast = attn_mask.to(q_x.dtype) if attn_mask is not None else None
            attn_out, attn_weights = self_block.attn(
                ln_q, k_x, v_x,
                need_weights=True,
                average_attn_weights=True,
                attn_mask=attn_mask_cast,
            )
            if attn_weights is not None:
                storage.append(attn_weights.detach().cpu())
            ls_1 = getattr(self_block, 'ls_1', nn.Identity())
            ls_2 = getattr(self_block, 'ls_2', nn.Identity())
            x = q_x + ls_1(attn_out)
            x = x + ls_2(self_block.mlp(self_block.ln_2(x)))
            return x

        for i, block in enumerate(self.model.farl_visual.transformer.resblocks):
            self._original_forwards[i] = block.forward
            block.forward = types.MethodType(_patched_forward_clip, block)

    def _restore_encoder_blocks_farl(self):
        for i, block in enumerate(self.model.farl_visual.transformer.resblocks):
            block.forward = self._original_forwards[i]
        self._original_forwards.clear()

    # ------------------------------------------------------------------
    # Rollout computation
    # ------------------------------------------------------------------

    def _rollout(self) -> torch.Tensor:
        # Stack: (num_layers, batch, seq, seq)
        attentions = torch.stack(self._attention_weights, dim=0)
        seq_len = attentions.shape[-1]
        eye = torch.eye(seq_len).unsqueeze(0).unsqueeze(0)

        # Add identity (residual connection) and re-normalize rows
        attentions = (attentions + eye) / 2
        attentions = attentions / attentions.sum(dim=-1, keepdim=True)

        # Discard lowest attention values to reduce noise
        threshold = attentions.flatten(-2).quantile(self.discard_ratio, dim=-1)
        threshold = threshold.unsqueeze(-1).unsqueeze(-1)
        attentions = torch.where(attentions >= threshold,
                                 attentions,
                                 torch.zeros_like(attentions))
        attentions = attentions / (attentions.sum(dim=-1, keepdim=True) + 1e-8)

        # Multiply across layers
        result = attentions[0]
        for i in range(1, attentions.shape[0]):
            result = torch.bmm(attentions[i], result)

        # CLS token's attention to patch tokens (exclude CLS-to-CLS at index 0)
        return result[:, 0, 1:]

    @torch.no_grad()
    def __call__(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Args:
            image_tensor: (1, C, H, W) on the same device as the model.

        Returns:
            rollout map as numpy array of shape (14, 14), values in [0, 1].
        """
        self._attention_weights.clear()
        self._original_forwards.clear()

        is_farl = getattr(self.model, 'backbone_type', 'imagenet_vit') == 'farl'

        if is_farl:
            self._patch_encoder_blocks_farl()
        else:
            self._patch_encoder_blocks()

        try:
            _ = self.model(image_tensor)
        finally:
            if is_farl:
                self._restore_encoder_blocks_farl()
            else:
                self._restore_encoder_blocks()

        if not self._attention_weights:
            raise RuntimeError("No attention weights captured after patching encoder blocks.")

        cls_attn = self._rollout()  # (1, 196)
        grid_size = int(cls_attn.shape[-1] ** 0.5)  # 14
        attn_map = cls_attn[0].reshape(grid_size, grid_size).numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        return attn_map


class DaViTGradCAM:
    """
    GradCAM visualization for DaViT backbone.

    Hooks the last stage output (B, C, H, W) — the spatial feature map before
    global average pooling — and computes a class activation map per emotion head.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._activations: Optional[torch.Tensor] = None

    def _target_layer(self) -> nn.Module:
        return self.model.davit.stages[-1]

    def compute(self, image_tensor: torch.Tensor, emotion_idx: int) -> np.ndarray:
        """
        Args:
            image_tensor: (1, C, H, W) on the same device as the model.
            emotion_idx: which emotion head (0-3) to visualize.

        Returns:
            CAM as numpy array of shape (H_stage, W_stage), values in [0, 1].
        """
        self._activations = None
        handle = None

        def fwd_hook(module, input, output):
            output.requires_grad_(True)
            output.retain_grad()
            self._activations = output

        handle = self._target_layer().register_forward_hook(fwd_hook)

        try:
            self.model.zero_grad()
            output = self.model(image_tensor)  # (1, num_emotions, num_classes)
            pred_class = output[0, emotion_idx].argmax().item()
            score = output[0, emotion_idx, pred_class]
            score.backward()
        finally:
            handle.remove()

        if self._activations is None or self._activations.grad is None:
            raise RuntimeError("GradCAM failed to capture activations or gradients.")

        acts = self._activations[0].detach()   # (C, H, W)
        grads = self._activations.grad[0].detach()  # (C, H, W)

        weights = grads.mean(dim=(1, 2))  # (C,)
        cam = (acts * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)  # (H, W)
        cam = torch.relu(cam).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


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
    Generates and saves attention/GradCAM heatmaps for `num_samples` images.

    For DaViT backbone: generates one GradCAM map per emotion head per image.
    For torchvision ViT / FaRL backbone: uses attention rollout (single map per image).

    Args:
        model: trained ViTEmotionModel.
        dataset: dataset instance (EmotionDataset or BinaryEmotionDataset).
        device: torch device.
        emotion_columns: list of emotion label names.
        output_dir: directory to save images.
        num_samples: how many samples to visualize.
        discard_ratio: fraction of lowest attention weights to zero out (rollout only).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    backbone_type = getattr(model, "backbone_type", "imagenet_vit")
    model.eval()

    indices = list(range(min(num_samples, len(dataset))))
    print(f"Generating {len(indices)} attention maps...")

    if backbone_type == "davit":
        gradcam = DaViTGradCAM(model)

        for idx in indices:
            image_tensor, labels = dataset[idx]
            image_tensor = image_tensor.unsqueeze(0).to(device)

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            display_tensor = image_tensor.squeeze(0).cpu() * std + mean
            display_tensor = display_tensor.clamp(0, 1)
            original_pil = Image.fromarray(
                (display_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )

            num_emotions = len(emotion_columns)
            ncols = 1 + num_emotions  # original + one cam per emotion
            fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 5))

            axes[0].imshow(original_pil)
            axes[0].set_title("Original")
            axes[0].axis("off")

            with torch.no_grad():
                preds = model(image_tensor).argmax(dim=2).squeeze(0).cpu().tolist()
            true_labels = labels.tolist()

            for emo_idx, emo_name in enumerate(emotion_columns):
                cam = gradcam.compute(image_tensor, emo_idx)
                overlay = overlay_attention_on_image(original_pil, cam)
                pred = preds[emo_idx]
                true = true_labels[emo_idx]
                axes[emo_idx + 1].imshow(overlay)
                axes[emo_idx + 1].set_title(f"{emo_name}\npred={pred} true={true}", fontsize=8)
                axes[emo_idx + 1].axis("off")

            plt.tight_layout()
            save_file = out_path / f"gradcam_sample_{idx:04d}.png"
            plt.savefig(save_file, dpi=120, bbox_inches="tight")
            plt.close(fig)

    else:
        rollout = AttentionRollout(model, discard_ratio=discard_ratio)

        for idx in indices:
            image_tensor, labels = dataset[idx]
            image_tensor = image_tensor.unsqueeze(0).to(device)

            attn_map = rollout(image_tensor)

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            display_tensor = image_tensor.squeeze(0).cpu() * std + mean
            display_tensor = display_tensor.clamp(0, 1)
            original_pil = Image.fromarray(
                (display_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )

            overlay = overlay_attention_on_image(original_pil, attn_map)

            with torch.no_grad():
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
            axes[1].set_title("Attention Map")
            axes[1].axis("off")

            axes[2].imshow(overlay)
            axes[2].set_title("Overlay")
            axes[2].axis("off")

            fig.suptitle(label_str, fontsize=9)
            plt.tight_layout()
            save_file = out_path / f"attention_sample_{idx:04d}.png"
            plt.savefig(save_file, dpi=120, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved {len(indices)} maps to: {out_path}")
