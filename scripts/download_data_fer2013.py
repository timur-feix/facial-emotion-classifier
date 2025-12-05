import kagglehub
from pathlib import Path
import shutil

target_path = Path("./data/raw")
preparation_processed_path = Path("./data/processed")

# download data from kaggle and retrieve destination
download_path = Path(kagglehub.dataset_download("msambare/fer2013"))

# copy the data into target path (./data/raw), also create (./data/processed)
target_path.mkdir(parents=True, exist_ok=True)
preparation_processed_path.mkdir(parents=True, exist_ok=True)
shutil.copytree(download_path, target_path, dirs_exist_ok=True)

# delete original downloaded data
shutil.rmtree(download_path)