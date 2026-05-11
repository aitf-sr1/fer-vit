from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

from .auxiliary_encoder import build_auxiliary_encoder


class ViTEmotionModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.num_emotions = config['model']['num_emotions']
        self.num_classes = config['model'].get('num_classes', 4)
        self.backbone_type = config['model'].get('backbone', 'imagenet_vit')

        if self.backbone_type == 'farl':
            in_features = self._build_farl_backbone(config)
        elif self.backbone_type == 'davit':
            in_features = self._build_davit_backbone(config)
        elif self.backbone_type == 'efficientvit':
            in_features = self._build_efficientvit_backbone(config)
        elif self.backbone_type == 'dinov3':
            in_features = self._build_dinov3_backbone(config)
        else:
            in_features = self._build_imagenet_backbone(config)

        self.auxiliary_encoder = build_auxiliary_encoder(config)
        head_in_features = in_features + (
            self.auxiliary_encoder.output_dim if self.auxiliary_encoder is not None else 0
        )

        self.emotion_heads = nn.ModuleList([
            nn.Linear(head_in_features, self.num_classes)
            for _ in range(self.num_emotions)
        ])

        if config['model']['freeze_backbone']:
            self.freeze_backbone()

    def _build_imagenet_backbone(self, config: Dict[str, Any]) -> int:
        if config['model']['pretrained']:
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.vit = vit_b_16(weights=None)

        original_head = self.vit.heads.head
        if not isinstance(original_head, nn.Linear):
            raise TypeError("Expected vit.heads.head to be nn.Linear")
        in_features = original_head.in_features

        self.vit.heads.head = nn.Identity()
        return in_features

    def _build_farl_backbone(self, config: Dict[str, Any]) -> int:
        import open_clip

        farl_checkpoint = config['model'].get('farl_checkpoint', None)
        clip_model = open_clip.create_model('ViT-B-16', pretrained=False)

        if farl_checkpoint is not None:
            state = torch.load(farl_checkpoint, map_location='cpu', weights_only=False)
            state_dict = state.get('state_dict', state)
            missing, unexpected = clip_model.load_state_dict(state_dict, strict=False)
            print(f"FaRL weights loaded from {farl_checkpoint}")
            print(f"  Missing keys:    {len(missing)}")
            print(f"  Unexpected keys: {len(unexpected)}")
        else:
            print("WARNING: farl_checkpoint not set. FaRL backbone initialised randomly.")

        self.farl_visual = clip_model.visual
        return clip_model.visual.output_dim

    def _build_davit_backbone(self, config: Dict[str, Any]) -> int:
        import timm

        model_name = config['model'].get('davit_variant', 'davit_base')
        pretrained = config['model'].get('pretrained', True)
        self.davit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        print(f"DaViT backbone: {model_name}, pretrained={pretrained}")
        return self.davit.num_features

    def _build_dinov3_backbone(self, config: Dict[str, Any]) -> int:
        import timm

        variant = config['model'].get('dinov3_variant', 'vit_small_patch16_dinov3')
        dinov3_checkpoint = config['model'].get('dinov3_checkpoint', None)
        self.dinov3 = timm.create_model(variant, pretrained=False, num_classes=0)

        if dinov3_checkpoint is not None:
            state = torch.load(dinov3_checkpoint, map_location='cpu', weights_only=False)
            state_dict = state.get('model', state.get('state_dict', state))
            missing, unexpected = self.dinov3.load_state_dict(state_dict, strict=False)
            print(f"DINOv3 weights loaded from {dinov3_checkpoint}")
            print(f"  Missing keys:    {len(missing)}")
            print(f"  Unexpected keys: {len(unexpected)}")
        else:
            print("WARNING: dinov3_checkpoint not set. DINOv3 backbone initialised randomly.")

        return self.dinov3.num_features

    def _build_efficientvit_backbone(self, config: Dict[str, Any]) -> int:
        import timm

        model_name = config['model'].get('efficientvit_variant', 'efficientvit_m2')
        pretrained = config['model'].get('pretrained', True)
        self.efficientvit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        print(f"EfficientViT backbone: {model_name}, pretrained={pretrained}")
        return self.efficientvit.num_features

    def freeze_backbone(self) -> None:
        if self.backbone_type == 'farl':
            for param in self.farl_visual.parameters():
                param.requires_grad = False
        elif self.backbone_type == 'davit':
            for param in self.davit.parameters():
                param.requires_grad = False
        elif self.backbone_type == 'efficientvit':
            for param in self.efficientvit.parameters():
                param.requires_grad = False
        elif self.backbone_type == 'dinov3':
            for param in self.dinov3.parameters():
                param.requires_grad = False
        else:
            for param in self.vit.conv_proj.parameters():
                param.requires_grad = False
            for param in self.vit.encoder.parameters():
                param.requires_grad = False
        for param in self.emotion_heads.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self) -> None:
        if self.backbone_type == 'farl':
            for param in self.farl_visual.parameters():
                param.requires_grad = True
        elif self.backbone_type == 'davit':
            for param in self.davit.parameters():
                param.requires_grad = True
        elif self.backbone_type == 'efficientvit':
            for param in self.efficientvit.parameters():
                param.requires_grad = True
        elif self.backbone_type == 'dinov3':
            for param in self.dinov3.parameters():
                param.requires_grad = True
        else:
            for param in self.vit.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor, aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.backbone_type == 'farl':
            features = self.farl_visual(x)
        elif self.backbone_type == 'davit':
            features = self.davit(x)
        elif self.backbone_type == 'efficientvit':
            features = self.efficientvit(x)
        elif self.backbone_type == 'dinov3':
            features = self.dinov3(x)
        else:
            features = self.vit(x)

        if self.auxiliary_encoder is not None and aux is not None:
            aux_features = self.auxiliary_encoder(aux)
            features = torch.cat([features, aux_features], dim=1)

        logits = torch.stack(
            [head(features) for head in self.emotion_heads], dim=1
        )
        return logits

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_model(config: Dict[str, Any]) -> ViTEmotionModel:
    return ViTEmotionModel(config)
