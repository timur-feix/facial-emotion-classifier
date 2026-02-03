import kagglehub as kh
import os
import shutil 
from pathlib import Path 

dest_dir = Path("data/balanced-raf-db")
dataset_path = kh.dataset_download("dollyprajapati182/balanced-raf-db-dataset-7575-grayscale")

dest_dir.mkdir(parents=True, exist_ok=True)

src_dir = Path(dataset_path)

for item in src_dir.iterdir():
    shutil.move(str(item), dest_dir / item.name)

shutil.rmtree(src_dir)