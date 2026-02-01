from pathlib import Path
from pandas import read_csv

import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset


INDEX_MAP = {"angry":0, "disgust":1, "fear":2, "happy":3, "sad":4, "surprise":5}


class RAFDataset(Dataset):
    def __init__(self, relative_pathname):
        self.relative_pathname = Path(relative_pathname)
        self.labels = read_csv(self.relative_pathname / "labels.csv")

        self.index_map = INDEX_MAP
        self.class_map = {v: k for k, v in self.index_map.items()}

        self.samples = []
        for _, row in self.labels.iterrows():
            label = row["label"]
            self.samples.append(
                (self.relative_pathname / row["filename"], self.index_map[label])
            )

    def __getitem__(self, idx):
        # implements getitem where each image is converted into a 64x64 "grey" RGB image
        path, y = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("L").convert("RGB")
            img = img.resize((64, 64), resample=Image.Resampling.BILINEAR)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))

        x = torch.from_numpy(arr)
        x = (x - 0.5) / 0.5
        y = torch.tensor(y, dtype=torch.long)

        return x, y
    
    def __len__(self):
        return len(self.samples)