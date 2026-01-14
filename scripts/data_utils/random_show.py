import random
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Paths
split_dir = Path("data/balanced-raf-db/train")
csv_path = split_dir / "labels.csv"

# Load labels
df = pd.read_csv(csv_path)

# Sample N random images
N = 24
samples = df.sample(n=N, random_state=None)

# Plot
plt.figure(figsize=(6, 3))

for i, (_, row) in enumerate(samples.iterrows()):
    img_path = split_dir / row["filename"]
    img = Image.open(img_path)

    plt.subplot(4, 6, i + 1)
    plt.imshow(img)
    plt.title(row["label"])
    plt.axis("off")

plt.tight_layout()
plt.show()
