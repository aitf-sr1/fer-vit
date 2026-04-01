from typing import Dict, Any
import torch.nn as nn


def create_model(config: Dict[str, Any]) -> nn.Module:
    model_name = config['model']['name']
    
    if model_name == 'graph_transformer':
        from .graph_transformer import create_model as create_graph_transformer
        return create_graph_transformer(config)
    elif model_name == 'vit_b_16':
        from .landmark_vit import create_model as create_vit
        return create_vit(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")
