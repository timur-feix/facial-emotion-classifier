import kagglehub as kh
import os
import shutil 
from pathlib import Path 

# Balanced Raf-Db

'''dest_dir = Path("data/balanced-raf-db")
dataset_path = kh.dataset_download("dollyprajapati182/balanced-raf-db-dataset-7575-grayscale")

dest_dir.mkdir(parents=True, exist_ok=True)

src_dir = Path(dataset_path)

for item in src_dir.iterdir():
    shutil.move(str(item), dest_dir / item.name)

shutil.rmtree(src_dir)

print("Balanced Raf-Db downloaded.")'''

# Fer2013

fer_dest = Path("data/fer2013")
fer_path = kh.dataset_download("msambare/fer2013")   

fer_dest.mkdir(parents=True, exist_ok=True)

for item in Path(fer_path).iterdir():
    shutil.move(str(item), fer_dest / item.name)

shutil.rmtree(fer_path)

print("FER2013 downloaded.")

# CK

ck_dest = Path("data/ckplus")


if ck_dest.exists():
    shutil.rmtree(ck_dest)

ck_path = kh.dataset_download("shuvoalok/ck-dataset")

ck_dest.mkdir(parents=True, exist_ok=True)

for item in Path(ck_path).iterdir():
    shutil.move(str(item), ck_dest / item.name)

shutil.rmtree(ck_path)

print("CK+ downloaded")
