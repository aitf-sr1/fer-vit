from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

from ..data.face_mesh import create_attention_bias


class LandmarkGuidedAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias_strength: float = 1.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.bias_strength = bias_strength
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        # Apply landmark bias if provided
        if attention_bias is not None:
            # attention_bias shape: [B, 196]
            # Add class token bias
            cls_bias = attention_bias.mean(dim=1, keepdim=True)  # [B, 1]
            bias_with_cls = torch.cat([cls_bias, attention_bias], dim=1)  # [B, 197]
            
            # Broadcast to [B, num_heads, N, N]
            # Each query attends to keys weighted by their landmark importance
            landmark_bias = bias_with_cls.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, 197]
            landmark_bias = landmark_bias.expand(B, self.num_heads, N, N)  # [B, num_heads, 197, 197]
            
            # Add bias to attention scores
            attn = attn + landmark_bias * self.bias_strength
        
        # Softmax and apply to values
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


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
            # Replace PyTorch's MultiheadAttention with our custom implementation
            original_attn = layer.self_attention
            if isinstance(original_attn, nn.MultiheadAttention):
                layer.self_attention = LandmarkGuidedAttention(
                    embed_dim=original_attn.embed_dim,
                    num_heads=original_attn.num_heads,
                    dropout=0.0,
                    bias_strength=self.bias_strength
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

        # Patch embedding
        x = self.vit.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # Add class token
        batch_class_token = self.vit.class_token.expand(batch_size, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # Add positional embedding
        x = self.vit.encoder.pos_embedding + x
        x = self.vit.encoder.dropout(x)

        # Pass through transformer layers with landmark bias
        for layer in self.vit.encoder.layers:
            # Layer norm + attention with landmark bias
            ln_out = layer.ln_1(x)
            attn_out = layer.self_attention(ln_out, attention_bias=attention_bias)
            x = x + layer.dropout(attn_out)
            
            # Layer norm + MLP
            x = x + layer.mlp(layer.ln_2(x))

        # Final layer norm
        x = self.vit.encoder.ln(x)

        # Classification head
        x = x[:, 0]
        x = self.vit.heads(x)

        return x

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_model(config: Dict[str, Any]) -> LandmarkGuidedTransformer:
    return LandmarkGuidedTransformer(config)
