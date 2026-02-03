import csv
import shutil
from pathlib import Path
from PIL import Image

DATA_ROOT = Path("data/balanced-raf-db")
IMAGE_SIZE = (64, 64)
# Removing 'neutral' class directories
for dir in ["train", "val", "test"]:
    shutil.rmtree(DATA_ROOT / dir / "neutral", ignore_errors=True)

def preprocess_split(split_dir):
    split_dir = Path(split_dir)
    label_rows = []

    for label_dir in sorted(split_dir.iterdir()):
        if not label_dir.is_dir():
            continue

        label = label_dir.name
        
        for idx, img_path in enumerate(sorted(label_dir.iterdir())):
            if not img_path.is_file():
                continue
            # loading images safely and resizing
            img = Image.open(img_path).convert("L")
            img = img.resize(IMAGE_SIZE)

            # Create unique filename
            new_name = f"{label}_{idx}_{img_path.name}"
            new_path = split_dir / new_name

            shutil.move(img_path, new_path)

            img.save(new_path)
            label_rows.append([new_name, label])
            

        # Remove empty label directory
        label_dir.rmdir()

    # Write labels.csv
    csv_path = split_dir / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(sorted(label_rows)) #sort for consistency

    print(f"Processed {split_dir}, saved {csv_path}")


# Run for each split
for split in ["train", "val", "test"]:
    preprocess_split(DATA_ROOT / split)