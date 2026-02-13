from pathlib import Path
from pandas import read_csv

import numpy as np
import torch
from PIL import Image


from torch.utils.data import Dataset

from src import utilities


INDEX_MAP = {"angry":0, "disgust":1, "fear":2, "happy":3, "sad":4, "surprise":5}


class RAFDataset(Dataset):
    def __init__(self, relative_pathname):
        self.relative_pathname = Path(relative_pathname)
        self.labels = read_csv(self.relative_pathname/ "labels.csv")

        self.index_map = INDEX_MAP
        self.class_map = {v: k for k, v in self.index_map.items()}

        self.samples = []
        for _, row in self.labels.iterrows():
            label = row["label"]
            self.samples.append(
                (self.relative_pathname / row["filename"], self.index_map[label])
            )
        
        self.timing_supervisor = [0.0, 0.0]
    
    def reset_timing_supervisor(self):
        self.timing_supervisor = [0.0, 0.0]

    def __getitem__(self, idx):
        with utilities.Timer("full __getitem__", show=False) as t:
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

        self.timing_supervisor[0] += t.timer
        self.timing_supervisor[1] += 1
        return x, y
    
    def __len__(self):
        return len(self.samples)
    
# dataset.py  (FER2013 folder-version)

from pathlib import Path
from typing import Optional, List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class FER2013FolderDataset(ImageFolder):
    

    def __init__(self, root: str, transform=None, drop_neutral: bool = False):
        super().__init__(root=root, transform=transform)

        if drop_neutral:
            self._drop_class("neutral")

    def _drop_class(self, class_name: str):
        
        if class_name not in self.class_to_idx:
            return

        drop_idx = self.class_to_idx[class_name]

       
        self.samples = [(p, y) for (p, y) in self.samples if y != drop_idx]
        self.imgs = self.samples  

        
        kept_classes = [c for c in self.classes if c != class_name]
        new_class_to_idx = {c: i for i, c in enumerate(kept_classes)}

        
        def remap(old_y: int) -> int:
            old_class = self.classes[old_y]
            return new_class_to_idx[old_class]

        self.samples = [(p, remap(y)) for (p, y) in self.samples]
        self.imgs = self.samples
        self.classes = kept_classes
        self.class_to_idx = new_class_to_idx


def get_fer2013_datasets(
    base_dir: str = "data/fer2013",
    transform=None,
    drop_neutral: bool = False,
) -> Tuple[FER2013FolderDataset, FER2013FolderDataset]:
    
    base = Path(base_dir)
    train_root = str(base / "train")
    test_root = str(base / "test")

    train_ds = FER2013FolderDataset(train_root, transform=transform, drop_neutral=drop_neutral)
    test_ds = FER2013FolderDataset(test_root, transform=transform, drop_neutral=drop_neutral)

    return train_ds, test_ds

class CKPlusDataset(ImageFolder):

    def __init__(self, root: str, transform=None, drop_contempt: bool = False):
        super().__init__(root=root, transform=transform)

        if drop_contempt:
            self._drop_class("contempt")

    def _drop_class(self, class_name: str):

        if class_name not in self.class_to_idx:
            return

        drop_idx = self.class_to_idx[class_name]

        
        self.samples = [(p, y) for (p, y) in self.samples if y != drop_idx]
        self.imgs = self.samples

        # class listesi güncelle
        kept_classes = [c for c in self.classes if c != class_name]
        new_class_to_idx = {c: i for i, c in enumerate(kept_classes)}

        def remap(old_y: int) -> int:
            old_class = self.classes[old_y]
            return new_class_to_idx[old_class]

        self.samples = [(p, remap(y)) for (p, y) in self.samples]
        self.imgs = self.samples
        self.classes = kept_classes
        self.class_to_idx = new_class_to_idx

def get_ckplus_datasets(base_dir: str, transform=None):
    return CKPlusDataset(
        root=base_dir,
        transform=transform,
        drop_contempt=True
    )

