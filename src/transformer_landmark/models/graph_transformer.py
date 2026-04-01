import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from typing import Dict, Any

from ..data.face_mesh import get_all_edges, NUM_LANDMARKS


class FacialLandmarkGraphTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        num_emotions: int = 4,
        dropout: float = 0.1,
        readout: str = 'mean',
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_emotions = num_emotions
        self.readout = readout
        
        edge_index, num_edges = get_all_edges()
        self.register_buffer('edge_index', edge_index)
        
        self.node_embedding = nn.Linear(2, d_model)
        self.pos_embedding = nn.Embedding(NUM_LANDMARKS, d_model)
        
        self.transformer_layers = nn.ModuleList([
            TransformerConv(
                in_channels=d_model,
                out_channels=d_model,
                heads=num_heads,
                dropout=dropout,
                concat=False,
            )
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_emotions)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
        batch_size = landmarks.shape[0]
        
        x = landmarks.view(-1, 2)
        x = self.node_embedding(x)
        
        pos_indices = torch.arange(NUM_LANDMARKS, device=landmarks.device)
        pos_indices = pos_indices.unsqueeze(0).repeat(batch_size, 1).view(-1)
        pos_emb = self.pos_embedding(pos_indices)
        
        x = x + pos_emb
        x = self.dropout(x)
        
        batch = torch.arange(batch_size, device=landmarks.device)
        batch = batch.unsqueeze(1).repeat(1, NUM_LANDMARKS).view(-1)
        
        edge_index_list = []
        for i in range(batch_size):
            offset = i * NUM_LANDMARKS
            edge_index_list.append(self.edge_index + offset)
        edge_index = torch.cat(edge_index_list, dim=1)
        
        for transformer, layer_norm in zip(self.transformer_layers, self.layer_norms):
            x_new = transformer(x, edge_index)
            x = layer_norm(x + x_new)
            x = self.dropout(x)
        
        if self.readout == 'mean':
            x = x.view(batch_size, NUM_LANDMARKS, self.d_model).mean(dim=1)
        elif self.readout == 'max':
            x = x.view(batch_size, NUM_LANDMARKS, self.d_model).max(dim=1)[0]
        elif self.readout == 'sum':
            x = x.view(batch_size, NUM_LANDMARKS, self.d_model).sum(dim=1)
        else:
            raise ValueError(f"Unknown readout: {self.readout}")
        
        emotions = self.classifier(x)
        
        return emotions
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: Dict[str, Any]) -> FacialLandmarkGraphTransformer:
    model_config = config['model']
    
    model = FacialLandmarkGraphTransformer(
        d_model=model_config.get('d_model', 128),
        num_heads=model_config.get('num_heads', 4),
        num_layers=model_config.get('num_layers', 4),
        num_emotions=model_config.get('num_emotions', 4),
        dropout=model_config.get('dropout', 0.1),
        readout=model_config.get('readout', 'mean'),
    )
    
    return model
