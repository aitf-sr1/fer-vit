from typing import Dict, Any
import torch
import torch.nn as nn


class LandmarkTransformer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.num_emotions = config["model"]["num_emotions"]
        
        self.num_landmarks = 468
        self.landmark_dim = 3
        
        embed_dim = config["model"].get("embed_dim", 256)
        num_heads = config["model"].get("num_heads", 8)
        num_layers = config["model"].get("num_layers", 6)
        dropout = config["model"].get("dropout", 0.1)
        
        self.landmark_embedding = nn.Linear(self.landmark_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_landmarks, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_emotions)
        
    def forward(self, x: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        batch_size = landmarks.shape[0]
        
        x = self.landmark_embedding(landmarks)
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        x = self.transformer(x)
        
        x = self.norm(x)
        x = x.mean(dim=1)
        
        x = self.head(x)
        
        return x
    
    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_model(config: Dict[str, Any]) -> LandmarkTransformer:
    return LandmarkTransformer(config)
