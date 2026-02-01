import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

import dataset
from model import FacialEmotionRecognitionCNN as FERCNN


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    loss_sum, correct, total = 0.0, 0, 0

    pbar = tqdm(
        loader,
        desc="Testing",
        dynamic_ncols=True,
        leave=True,
    )

    for x, y in pbar:
        x = x.to(device, non_blocking=(device.type == "cuda"))
        y = y.to(device, non_blocking=(device.type == "cuda"))

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        loss_sum += loss.item() * bs

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += bs

        pbar.set_postfix(
            acc=f"{correct/total:.3f}",
            loss=f"{loss.item():.3f}",
        )

    acc = correct / total
    avg_loss = loss_sum / total
    return acc, avg_loss


def main():
    root = Path(__file__).resolve().parents[1]

    data_dir = root / "data" / "balanced-raf-db" / "test"
    ckpt_path = root / "checkpoints" / "best.pt"

    if not data_dir.exists():
        raise FileNotFoundError(f"Test split not found: {data_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    model = FERCNN().to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    test_ds = dataset.RAFDataset(str(data_dir))
    test_ld = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    acc, loss = evaluate(model, test_ld, device)
    print(f"TEST: Accuracy {acc:.3f}; Loss {loss:.3f}", flush=True)


if __name__ == "__main__":
    main()
