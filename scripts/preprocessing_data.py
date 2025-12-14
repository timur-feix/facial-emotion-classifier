import csv
import shutil
from pathlib import Path
from PIL import Image

for dir in ["train", "val", "test"]:
    shutil.rmtree(f"data/balanced-raf-db/{dir}/neutral", ignore_errors=True)

def flatten_split(split_dir):
    split_dir = Path(split_dir)
    label_rows = []

    for label_dir in split_dir.iterdir():
        if not label_dir.is_dir():
            continue

        label = label_dir.name

        for idx, img_path in enumerate(label_dir.iterdir()):
            if not img_path.is_file():
                continue

            # Create unique filename
            new_name = f"{label}_{idx}_{img_path.name}"
            new_path = split_dir / new_name

            shutil.move(img_path, new_path)

            label_rows.append([new_name, label])

        # Remove empty label directory
        label_dir.rmdir()

    # Write labels.csv
    csv_path = split_dir / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        writer.writerows(label_rows)

    print(f"Processed {split_dir}, saved {csv_path}")


# Run for each split
for split in ["train", "val", "test"]:
    flatten_split(f"data/balanced-raf-db/{split}")



# we converted the images to RGB
def convert_images_to_rgb(split_dir):
    split_dir = Path(split_dir)

    for img_path in split_dir.iterdir():
        if not img_path.is_file():
            continue

        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(img_path)

    print(f"Converted images to RGB in {split_dir}")

for split in ["train", "val", "test"]:
    convert_images_to_rgb(f"data/balanced-raf-db/{split}")



