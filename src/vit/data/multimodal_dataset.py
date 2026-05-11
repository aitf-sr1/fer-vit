from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..models.auxiliary_encoder import LandmarkEncoder


class MultiModalEmotionDataset(Dataset):
    EMOTION_COLUMNS = ["Boredom", "Engagement", "Confusion", "Frustration"]
    NUM_LANDMARKS = 478

    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        transform: Callable,
        aux_type: str,
        image_path_column: str = "Image_Name",
        aux_csv: Optional[str] = None,
        au_columns: Optional[List[str]] = None,
    ):
        self.data = pd.read_csv(csv_file)
        if aux_csv is not None:
            aux_data = pd.read_csv(aux_csv)
            self.data = pd.concat(
                [self.data.reset_index(drop=True), aux_data.reset_index(drop=True)],
                axis=1,
            )

        self.img_dir = Path(img_dir)
        self.transform = transform
        self.aux_type = aux_type
        self.image_path_column = image_path_column
        self.au_columns: List[str] = au_columns or []

        if aux_type in ('mediapipe_landmarks', 'both'):
            landmark_cols = [
                f"landmark_{i}_{axis}"
                for i in range(self.NUM_LANDMARKS)
                for axis in ("x", "y")
            ]
            missing = [c for c in landmark_cols if c not in self.data.columns]
            if missing:
                raise ValueError(
                    f"CSV missing landmark columns. First missing: {missing[0]}"
                )
            self._landmark_cols = landmark_cols

        if aux_type in ('action_units', 'both'):
            if not self.au_columns:
                raise ValueError(
                    "aux_type includes 'action_units' but au_columns is empty."
                )
            missing = [c for c in self.au_columns if c not in self.data.columns]
            if missing:
                raise ValueError(
                    f"CSV missing AU columns. First missing: {missing[0]}"
                )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = int(idx.item())

        row = self.data.iloc[idx]

        img_path = self.img_dir / row[self.image_path_column]
        image = self.transform(Image.open(img_path).convert("RGB"))

        labels = torch.tensor(
            [int(row[col]) for col in self.EMOTION_COLUMNS],
            dtype=torch.long,
        )

        aux = self._build_aux_tensor(row)

        return image, aux, labels

    def _build_aux_tensor(self, row: pd.Series) -> torch.Tensor:
        parts = []

        if self.aux_type in ('mediapipe_landmarks', 'both'):
            lm = np.array([row[c] for c in self._landmark_cols], dtype=np.float32)
            lm = self._normalize_landmarks(lm)
            parts.append(torch.from_numpy(lm))

        if self.aux_type in ('action_units', 'both'):
            au = np.array([float(row[c]) for c in self.au_columns], dtype=np.float32)
            parts.append(torch.from_numpy(au))

        return torch.cat(parts)

    @staticmethod
    def _normalize_landmarks(lm: np.ndarray) -> np.ndarray:
        coords = lm.reshape(-1, 2)
        coords = coords - coords.mean(axis=0)
        scale = coords.std()
        if scale > 0:
            coords = coords / scale
        return coords.flatten()

    def get_num_emotions(self) -> int:
        return len(self.EMOTION_COLUMNS)

    def aux_dim(self) -> int:
        dim = 0
        if self.aux_type in ('mediapipe_landmarks', 'both'):
            dim += LandmarkEncoder.INPUT_DIM
        if self.aux_type in ('action_units', 'both'):
            dim += len(self.au_columns)
        return dim
