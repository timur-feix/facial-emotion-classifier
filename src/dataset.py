from pathlib import Path
from pandas import read_csv

import numpy as np
import torch
from PIL import Image


INDEX_MAP = {"angry":0, "disgust":1, "fear":2, "happy":3, "sad":4, "surprise":5}


class BalancedRafDbDataset:
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
        path, y = self.samples[idx]
        PIL_image = Image.open(path)
        PIL_image = PIL_image.resize((64, 64), resample=Image.Resampling.BILINEAR)
                                     

        arr = np.array(PIL_image, dtype=np.float32) / 255.0
        arr = arr[None, :, :]
        arr = np.repeat(arr, 3, axis=0)

        x = torch.from_numpy(arr)
        y = torch.tensor(y, dtype=torch.long)

        return x, y
    
    def __len__(self):
        return len(self.samples)
    