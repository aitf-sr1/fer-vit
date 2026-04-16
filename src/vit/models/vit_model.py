from typing import Dict, Any
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTEmotionModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.num_emotions = config['model']['num_emotions']
        self.num_classes = config['model']['num_classes']

        if config['model']['pretrained']:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.vit = vit_b_16(weights=weights)
        else:
            self.vit = vit_b_16(weights=None)

        original_head = self.vit.heads.head
        if isinstance(original_head, nn.Linear):
            in_features = original_head.in_features
        else:
            raise TypeError("Expected vit.heads.head to be nn.Linear")

        # Replace classification head with identity; emotion heads are separate.
        self.vit.heads.head = nn.Identity()
        self.emotion_heads = nn.ModuleList([
            nn.Linear(in_features, self.num_classes)
            for _ in range(self.num_emotions)
        ])

        if config['model']['freeze_backbone']:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.vit.conv_proj.parameters():
            param.requires_grad = False
        for param in self.vit.encoder.parameters():
            param.requires_grad = False
        for param in self.emotion_heads.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        for param in self.vit.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.vit(x)  # (batch, in_features)
        logits = torch.stack(
            [head(features) for head in self.emotion_heads], dim=1
        )
        return logits  # (batch, num_emotions, num_classes)

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_model(config: Dict[str, Any]) -> ViTEmotionModel:
    return ViTEmotionModel(config)
