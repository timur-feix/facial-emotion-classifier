import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.model import FacialEmotionRecognitionCNN as FERCNN
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "surprise"]

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def pil_to_tensor_rgb_like(img: Image.Image) -> torch.Tensor:
    img = img.convert("L")

    try:
        resample = Image.Resampling.BILINEAR
    except AttributeError:
        resample = Image.BILINEAR # type: ignore

    img = img.resize((64, 64), resample=resample)

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr[None, :, :]
    arr = np.repeat(arr, 3, axis=0)
    return torch.from_numpy(arr)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Folder containing images to classify",
                        dest="input_dir")
    parser.add_argument("--weights", type=str, default="checkpoints/best.pt",
                        help="Model weights")
    parser.add_argument("--output-csv", type=str, default="predictions.csv",
                        help="Output CSV path (default: predictions.csv)",
                        dest="output_csv")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for inference (default: 256)",
                        dest="batch_size")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input dir not found or not a directory: {input_dir}")

    image_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if len(image_paths) == 0:
        raise SystemExit(f"No images found in {input_dir} with extensions {sorted(IMG_EXTS)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FERCNN().to(device)
    state = torch.load(args.weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]

    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    out_path = Path(args.output_csv)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"] + [f"score_{c}" for c in CLASSES])

        bs = args.batch_size
        for start in range(0, len(image_paths), bs):
            batch_paths = image_paths[start:start + bs]

            xs = []
            for p in batch_paths:
                img = Image.open(p)
                x = pil_to_tensor_rgb_like(img)
                xs.append(x)

            X = torch.stack(xs, dim=0).to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)

            probs = probs.cpu().numpy()
            for p, row in zip(batch_paths, probs):
                writer.writerow([p.name] + [float(v) for v in row])

    print(f"Wrote: {out_path} ({len(image_paths)} images) on device={device}")


if __name__ == "__main__":
    main()
