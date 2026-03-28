from typing import Dict, Any
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViTEmotionModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.num_emotions = config['model']['num_emotions']
        
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
        
        self.vit.heads.head = nn.Linear(in_features, self.num_emotions)
        
        if config['model']['freeze_backbone']:
            self.freeze_backbone()
    
    def freeze_backbone(self) -> None:
        for param in self.vit.conv_proj.parameters():
            param.requires_grad = False
        for param in self.vit.encoder.parameters():
            param.requires_grad = False
        
        for param in self.vit.heads.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self) -> None:
        for param in self.vit.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(x)
    
    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_model(config: Dict[str, Any]) -> ViTEmotionModel:
    return ViTEmotionModel(config)
