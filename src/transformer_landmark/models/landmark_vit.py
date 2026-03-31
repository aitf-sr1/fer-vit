from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

from ..data.face_mesh import create_attention_bias


class LandmarkGuidedAttention(nn.Module):
    def __init__(
        self, original_attention: nn.MultiheadAttention, bias_strength: float = 1.0
    ):
        super().__init__()
        self.attention = original_attention
        self.bias_strength = bias_strength

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if attention_bias is not None and attn_mask is None:
            batch_size = query.shape[1]
            seq_len = query.shape[0]
            num_heads = self.attention.num_heads

            bias = attention_bias.unsqueeze(1).expand(-1, seq_len, -1)
            bias = bias * self.bias_strength

            bias = bias.unsqueeze(1).expand(-1, num_heads, -1, -1)
            bias = bias.reshape(batch_size * num_heads, seq_len, seq_len)

            attn_mask = bias

        return self.attention(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
        )


class LandmarkGuidedTransformer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.num_emotions = config["model"]["num_emotions"]

        landmark_config = config.get("landmark_settings", {})
        self.image_size = config["data"]["image_size"]
        self.patch_size = 16
        self.sigma = landmark_config.get("sigma", 10.0)
        self.bias_strength = landmark_config.get("bias_strength", 1.0)

        if config["model"]["pretrained"]:
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

        self._inject_landmark_attention()

        if config["model"]["freeze_backbone"]:
            self.freeze_backbone()

    def _inject_landmark_attention(self) -> None:
        for _, layer in enumerate(self.vit.encoder.layers):
            original_attn = layer.self_attention
            if not isinstance(original_attn, nn.MultiheadAttention):
                raise TypeError("Expected layer.self_attention to be nn.MultiheadAttention")
            layer.self_attention = LandmarkGuidedAttention(
                original_attn, bias_strength=self.bias_strength
            )

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

    def forward(self, x: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        device = x.device

        attention_biases = []
        for i in range(batch_size):
            landmark_np = landmarks[i].cpu().numpy()
            bias = create_attention_bias(
                landmark_np,
                image_size=self.image_size,
                patch_size=self.patch_size,
                sigma=self.sigma,
            )
            attention_biases.append(bias)

        attention_bias = torch.stack(attention_biases).to(device)

        x = self.vit.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        batch_class_token = self.vit.class_token.expand(batch_size, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder.pos_embedding + x
        x = self.vit.encoder.dropout(x)

        x = x.transpose(0, 1)

        for layer in self.vit.encoder.layers:
            ln_1 = layer.ln_1
            assert isinstance(ln_1, nn.Module)
            ln_1_out = ln_1(x)

            self_attn = layer.self_attention
            assert isinstance(self_attn, LandmarkGuidedAttention)
            attn_out, _ = self_attn(
                ln_1_out,
                ln_1_out,
                ln_1_out,
                attention_bias=attention_bias,
                need_weights=False,
            )
            
            dropout = layer.dropout
            assert isinstance(dropout, nn.Module)
            x = x + dropout(attn_out)

            ln_2 = layer.ln_2
            assert isinstance(ln_2, nn.Module)
            mlp = layer.mlp
            assert isinstance(mlp, nn.Module)
            x = x + mlp(ln_2(x))

        x = x.transpose(0, 1)

        x = self.vit.encoder.ln(x)

        x = x[:, 0]

        x = self.vit.heads(x)

        return x

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_model(config: Dict[str, Any]) -> LandmarkGuidedTransformer:
    return LandmarkGuidedTransformer(config)
