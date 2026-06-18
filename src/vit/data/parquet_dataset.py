from io import BytesIO
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ParquetEmotionDataset(Dataset):
    EMOTION_COLUMNS = ["Boredom", "Engagement", "Confusion", "Frustration"]

    def __init__(
        self,
        parquet_path: str,
        transform: Callable,
        filter_synthetic: bool = False,
    ):
        self.data = pd.read_parquet(parquet_path, engine="pyarrow")
        self.transform = transform
        self.parquet_path = Path(parquet_path)

        if filter_synthetic and "synthetic" in self.data.columns:
            self.data = self.data[self.data["synthetic"] == 0].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = int(idx.item())

        row = self.data.iloc[idx]
        img_bytes = row["image"]["bytes"]
        image = Image.open(BytesIO(img_bytes)).convert("RGB")

        labels = torch.tensor(
            [int(row[col]) for col in self.EMOTION_COLUMNS],
            dtype=torch.long,
        )

        image = self.transform(image)

        return image, labels

    def get_num_emotions(self) -> int:
        return len(self.EMOTION_COLUMNS)
