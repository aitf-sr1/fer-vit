from io import BytesIO
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import pyarrow.parquet as pq
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
        self.parquet_path = str(parquet_path)
        self.transform = transform
        self.filter_synthetic = filter_synthetic

        pf = pq.ParquetFile(self.parquet_path)
        self._length = pf.metadata.num_rows

        self._data: Optional[pd.DataFrame] = None

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = int(idx.item())

        if self._data is None:
            df = pd.read_parquet(self.parquet_path, engine="pyarrow")
            if self.filter_synthetic and "synthetic" in df.columns:
                df = df[df["synthetic"] == 0].reset_index(drop=True)
            self._data = df

        row = self._data.iloc[idx]
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
