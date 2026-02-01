import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

import dataset
from model import FacialEmotionRecognitionCNN as FERCNN


@torch.no_grad()
def collect_confusion(model, loader, device, num_classes: int):
    model.eval()
    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)  # rows=true, cols=pred

    pbar = tqdm(loader, desc="Testing", dynamic_ncols=True, leave=True)
    for x, y in pbar:
        x = x.to(device, non_blocking=(device.type == "cuda"))
        y = y.to(device, non_blocking=(device.type == "cuda"))

        logits = model(x)
        preds = logits.argmax(dim=1)

        # accumulate confusion counts
        for t, p in zip(y.view(-1), preds.view(-1)):
            conf[int(t), int(p)] += 1

        total = conf.sum().item()
        correct = conf.diag().sum().item()
        pbar.set_postfix(acc=f"{correct/total:.3f}")

    return conf


def print_confusion(conf: torch.Tensor, class_map: dict[int, str], topk: int = 10):
    num_classes = conf.shape[0]
    names = [class_map[i] for i in range(num_classes)]

    # overall acc
    total = conf.sum().item()
    correct = conf.diag().sum().item()
    acc = correct / total if total > 0 else 0.0

    # normalized by true class (rows)
    conf_f = conf.to(torch.float32)
    row_sums = conf_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    conf_norm = conf_f / row_sums  # each row sums to 1

    # pretty print normalized matrix (%)
    header = "true\\pred | " + " ".join([f"{n[:3]:>5}" for n in names])
    print("\n" + header)
    print("-" * len(header))

    for i, name in enumerate(names):
        row = " ".join([f"{(100.0*conf_norm[i, j]):5.1f}" for j in range(num_classes)])
        print(f"{name[:9]:>9} | {row}")

    print(f"\nOverall accuracy: {acc:.3f}  (correct={correct} / total={total})")

    # Top confusions (excluding diagonal)
    confusions = []
    for t in range(num_classes):
        for p in range(num_classes):
            if t == p:
                continue
            c = int(conf[t, p].item())
            if c > 0:
                confusions.append((c, t, p))

    confusions.sort(reverse=True)
    print(f"\nTop {topk} confusions (count, true -> pred):")
    for c, t, p in confusions[:topk]:
        true_name = names[t]
        pred_name = names[p]
        row_total = int(conf[t].sum().item())
        pct = (c / row_total) * 100.0 if row_total > 0 else 0.0
        print(f"  {c:4d}  ({pct:5.1f}%)  {true_name} -> {pred_name}")


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
    class_map = test_ds.class_map  # {idx: label_name}
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

    conf = collect_confusion(model, test_ld, device, num_classes)
    print_confusion(conf, class_map, topk=10)


if __name__ == "__main__":
    main()
