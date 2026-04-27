from typing import Dict, Any
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


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
        else:
            in_features = self._build_imagenet_backbone(config)

        self.emotion_heads = nn.ModuleList([
            nn.Linear(in_features, self.num_classes)
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
        return clip_model.visual.output_dim  # 512 for ViT-B/16

    def _build_davit_backbone(self, config: Dict[str, Any]) -> int:
        import timm

        model_name = config['model'].get('davit_variant', 'davit_base')
        pretrained = config['model'].get('pretrained', True)
        self.davit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        print(f"DaViT backbone: {model_name}, pretrained={pretrained}")
        return self.davit.num_features

    def freeze_backbone(self) -> None:
        if self.backbone_type == 'farl':
            for param in self.farl_visual.parameters():
                param.requires_grad = False
        elif self.backbone_type == 'davit':
            for param in self.davit.parameters():
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
        else:
            for param in self.vit.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_type == 'farl':
            features = self.farl_visual(x)  # (batch, 512)
        elif self.backbone_type == 'davit':
            features = self.davit(x)        # (batch, 1024)
        else:
            features = self.vit(x)           # (batch, 768)

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
