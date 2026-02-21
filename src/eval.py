import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

import src.dataset as dataset
from src.model import FacialEmotionRecognitionCNN as FERCNN


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    loss_sum = 0.0
    correct = 0
    total = 0

    class_correct = [0] * num_classes
    class_total = [0] * num_classes

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

        # per-class stats
        for cls in range(num_classes):
            mask = (y == cls)
            n = mask.sum().item()
            if n == 0:
                continue
            class_total[cls] += n
            class_correct[cls] += (preds[mask] == cls).sum().item()

        pbar.set_postfix(
            acc=f"{correct/total:.3f}",
            loss=f"{loss.item():.3f}",
        )

    acc = correct / total
    avg_loss = loss_sum / total
    class_acc = [
        (class_correct[i] / class_total[i]) if class_total[i] > 0 else 0.0
        for i in range(num_classes)
    ]

    return acc, avg_loss, class_acc, class_total


def main():
    root = Path(__file__).resolve().parents[1]

    test_dir = root / "data" / "balanced-raf-db" / "test"
    ckpt_path = root / "checkpoints" / "best.pt"

    if not test_dir.exists():
        raise FileNotFoundError(f"Test split not found: {test_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    # model
    model = FERCNN().to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    # dataset
    test_ds = dataset.RAFDataset(str(test_dir))

    # class mapping (index -> label name)
    # class_map is {index: label_name}
    class_map = test_ds.class_map
    num_classes = len(class_map)

    test_ld = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
        prefetch_factor=2,
    )

    acc, loss, class_acc, class_total = evaluate(
        model, test_ld, device, num_classes
    )

    print(f"\nTEST: Accuracy {acc:.3f}; Loss {loss:.3f}", flush=True)
    print("Per-class accuracy:", flush=True)

    for idx in range(num_classes):
        name = class_map[idx]
        n = class_total[idx]
        a = class_acc[idx]
        print(f"  {idx:>2} | {name:<10} | acc {a:.3f} | n={n}", flush=True)


if __name__ == "__main__":
    main()
