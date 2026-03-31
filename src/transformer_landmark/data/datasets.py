from pathlib import Path
from typing import Callable, cast

import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .face_mesh import normalize_landmarks


class LandmarkEmotionDataset(Dataset):
    EMOTION_COLUMNS = ["Boredom", "Engagement", "Confusion", "Frustration"]
    NUM_LANDMARKS = 468

    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        transform: Callable,
    ):
        self.data = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Verify landmark columns exist
        landmark_cols = [f"landmark_{i}_{axis}" for i in range(self.NUM_LANDMARKS) for axis in ['x', 'y']]
        missing_cols = [col for col in landmark_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"CSV missing landmark columns. First missing: {missing_cols[0]}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = int(idx.item())

        row = self.data.iloc[idx]
        img_name = row["Image_Name"]
        img_path = self.img_dir / img_name

        image = Image.open(img_path).convert("RGB")
        labels = torch.tensor(
            [row[col] for col in self.EMOTION_COLUMNS],
            dtype=torch.float32,
        )

        # Load precomputed landmarks from CSV columns
        landmark_data = []
        for i in range(self.NUM_LANDMARKS):
            x = float(row[f"landmark_{i}_x"])
            y = float(row[f"landmark_{i}_y"])
            z = 0.0  # No z-coordinate in precomputed data
            landmark_data.append([x, y, z])
        
        landmarks_np = np.array(landmark_data, dtype=np.float32)
        
        # Normalize landmarks
        landmarks_np = normalize_landmarks(landmarks_np)

        landmarks = torch.from_numpy(landmarks_np).float()

        image = self.transform(image)

        return image, landmarks, labels

    def get_num_emotions(self) -> int:
        return len(self.EMOTION_COLUMNS)

    def get_emotion_statistics(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self.data[self.EMOTION_COLUMNS].describe())
