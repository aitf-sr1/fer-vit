from pathlib import Path
from typing import Callable, cast

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class EmotionDataset(Dataset):
    EMOTION_COLUMNS = ["Boredom", "Engagement", "Confusion", "Frustration"]

    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        transform: Callable,
    ):
        self.data = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = int(idx.item())

        img_name = self.data.iloc[idx]["Image_Name"]
        img_path = self.img_dir / img_name

        image = Image.open(img_path).convert("RGB")
        labels = torch.tensor(
            [self.data.iloc[idx][col] for col in self.EMOTION_COLUMNS],
            dtype=torch.float32,
        )

        image = self.transform(image)

        return image, labels

    def get_num_emotions(self) -> int:
        return len(self.EMOTION_COLUMNS)

    def get_emotion_statistics(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self.data[self.EMOTION_COLUMNS].describe())
