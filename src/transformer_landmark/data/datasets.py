from typing import Callable, cast, Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from .face_mesh import normalize_landmarks_2d


class LandmarkEmotionDataset(Dataset):
    EMOTION_COLUMNS = ["Boredom", "Engagement", "Confusion", "Frustration"]
    NUM_LANDMARKS = 478

    def __init__(
        self,
        csv_file: str,
        transform: Optional[Callable] = None,
        mode: str = "classification",
    ):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.mode = mode

        landmark_cols = [
            f"landmark_{i}_{axis}"
            for i in range(self.NUM_LANDMARKS)
            for axis in ["x", "y"]
        ]
        missing_cols = [col for col in landmark_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(
                f"CSV missing landmark columns. First missing: {missing_cols[0]}"
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = int(idx.item())

        row = self.data.iloc[idx]

        if self.mode == "classification":
            labels = torch.tensor(
                [int(row[col]) for col in self.EMOTION_COLUMNS],
                dtype=torch.long,
            )
        else:
            labels = torch.tensor(
                [row[col] for col in self.EMOTION_COLUMNS],
                dtype=torch.float32,
            )

        landmark_data = []
        for i in range(self.NUM_LANDMARKS):
            x = float(row[f"landmark_{i}_x"])
            y = float(row[f"landmark_{i}_y"])
            landmark_data.append([x, y])

        landmarks_np = np.array(landmark_data, dtype=np.float32)
        landmarks_np = normalize_landmarks_2d(landmarks_np)
        landmarks = torch.from_numpy(landmarks_np).float()

        if self.transform is not None:
            landmarks = self.transform(landmarks)

        return landmarks, labels

    def get_num_emotions(self) -> int:
        return len(self.EMOTION_COLUMNS)

    def get_emotion_statistics(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self.data[self.EMOTION_COLUMNS].describe())
