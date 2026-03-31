from pathlib import Path
from typing import Callable, cast

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .face_mesh import (
    FaceMeshExtractor,
    normalize_landmarks,
    create_zero_landmarks,
)


class LandmarkEmotionDataset(Dataset):
    EMOTION_COLUMNS = ["Boredom", "Engagement", "Confusion", "Frustration"]

    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        transform: Callable,
        fail_strategy: str = "zeros",
    ):
        self.data = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.fail_strategy = fail_strategy
        self.extractor = FaceMeshExtractor()
        
        self.failed_extractions = 0
        self.total_extractions = 0

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = int(idx.item())

        img_name = self.data.iloc[idx]["Image_Name"]
        img_path = self.img_dir / img_name

        image = Image.open(img_path).convert("RGB")
        labels = torch.tensor(
            [self.data.iloc[idx][col] for col in self.EMOTION_COLUMNS],
            dtype=torch.float32,
        )

        landmarks_np = self.extractor.extract_landmarks(image)
        
        self.total_extractions += 1
        if landmarks_np is None:
            self.failed_extractions += 1
            if self.fail_strategy == "zeros":
                landmarks_np = create_zero_landmarks()
            elif self.fail_strategy == "skip":
                return self.__getitem__((idx + 1) % len(self))
            else:
                raise ValueError(
                    f"Face mesh extraction failed for {img_name}. "
                    f"Set fail_strategy='zeros' or 'skip'."
                )
        else:
            landmarks_np = normalize_landmarks(landmarks_np)

        landmarks = torch.from_numpy(landmarks_np).float()

        image = self.transform(image)

        return image, landmarks, labels

    def get_num_emotions(self) -> int:
        return len(self.EMOTION_COLUMNS)

    def get_emotion_statistics(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self.data[self.EMOTION_COLUMNS].describe())

    def get_failure_rate(self) -> float:
        if self.total_extractions == 0:
            return 0.0
        return self.failed_extractions / self.total_extractions
