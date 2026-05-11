from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class LandmarkEncoder(nn.Module):
    INPUT_DIM = 478 * 2

    def __init__(self, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.flatten(start_dim=1)
        return self.net(x)


class AUEncoder(nn.Module):
    def __init__(self, num_au: int, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_au, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AuxiliaryEncoder(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        aux_cfg = config['model']['auxiliary']
        self.aux_type: str = aux_cfg['type']
        embed_dim: int = aux_cfg.get('embed_dim', 128)
        dropout: float = aux_cfg.get('dropout', 0.1)

        self.landmark_encoder: Optional[LandmarkEncoder] = None
        self.au_encoder: Optional[AUEncoder] = None

        if self.aux_type in ('mediapipe_landmarks', 'both'):
            self.landmark_encoder = LandmarkEncoder(embed_dim=embed_dim, dropout=dropout)

        if self.aux_type in ('action_units', 'both'):
            num_au = len(config['data'].get('au_columns', []))
            if num_au == 0:
                raise ValueError(
                    "aux_type includes 'action_units' but data.au_columns is empty."
                )
            self.au_encoder = AUEncoder(num_au=num_au, embed_dim=embed_dim, dropout=dropout)

        n_active = sum([
            self.landmark_encoder is not None,
            self.au_encoder is not None,
        ])
        self._output_dim = embed_dim * n_active

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, aux: torch.Tensor) -> torch.Tensor:
        parts = []

        if self.landmark_encoder is not None and self.au_encoder is not None:
            lm_dim = LandmarkEncoder.INPUT_DIM
            parts.append(self.landmark_encoder(aux[:, :lm_dim]))
            parts.append(self.au_encoder(aux[:, lm_dim:]))
        elif self.landmark_encoder is not None:
            parts.append(self.landmark_encoder(aux))
        elif self.au_encoder is not None:
            parts.append(self.au_encoder(aux))

        return torch.cat(parts, dim=1)


def build_auxiliary_encoder(config: Dict[str, Any]) -> Optional[AuxiliaryEncoder]:
    aux_cfg = config.get('model', {}).get('auxiliary', {})
    if not aux_cfg.get('enabled', False):
        return None
    return AuxiliaryEncoder(config)
