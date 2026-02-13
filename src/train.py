from src import utilities

import torch, os, argparse, json, socket

from .model import FacialEmotionRecognitionCNN
# from .dataset import get_fer2013_datasets
from .dataset import get_ckplus_datasets

from torch.utils.data import random_split
from torch.utils.data import DataLoader

from pathlib import Path
from torchvision import transforms

using_debug = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--outdir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--resume", type=str, default=None)
    return p.parse_args()


def save_ckpt(path, epoch, model, optimizer, best_valid_acc):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_valid_acc": best_valid_acc,
    }, path)


def load_ckpt(path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_valid_acc", 0.0)


def main():
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    default_data = root / "data" / "balanced-raf-db"

    data_dir = Path(args.data_dir) if args.data_dir else default_data

    outdir = (root / args.outdir) if not Path(args.outdir).is_absolute() else Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(vars(args), indent=2))

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    num_workers = args.num_workers

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
        print("CUDA:", torch.version.cuda, flush=True)

    model = FacialEmotionRecognitionCNN(n_classes=6).to(device)

    pin_memory = (device.type == "cuda")
    persistent_workers = (num_workers > 0)

    # loss
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    if using_debug:
        print(f"Using optim: {optimizer}", flush=True)

    best_valid_acc = 0.0
    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = root / resume_path
        start_epoch, best_valid_acc = load_ckpt(resume_path, model, optimizer)

    # datasets
    dl_common = dict(
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    #Rafdb
    '''train_ds = dataset.RAFDataset(str(data_dir / "train"))
    valid_ds = dataset.RAFDataset(str(data_dir / "val"))'''

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    #fer2013
    '''
    train_ds, valid_ds = get_fer2013_datasets(
    base_dir=str(root / "data" / "fer2013"),
    transform=transform,
    drop_neutral=True)'''

    # ckplus
    dataset = get_ckplus_datasets(
        base_dir=str(root / "data" / "ckplus"),
        transform=transform
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, valid_ds = random_split(dataset, [train_size, val_size])

    print("Train:", len(train_ds))
    print("Val:", len(valid_ds))

    if num_workers > 0:
        dl_common["prefetch_factor"] = 2

    train_ld = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_common)
    valid_ld = DataLoader(valid_ds, shuffle=False, drop_last=False, **dl_common)

    def do_epoch(mode):

        if mode == "train":
            context_manager = utilities.NullContext()
            model.train()
            data_loader = train_ld
        else:
            if mode == "valid":
                context_manager = torch.no_grad()
                model.eval()
                data_loader = valid_ld
            else:
                return 0, 0

        loss_sum, correct, total = 0, 0, 0
        with context_manager:
            for x, y in data_loader:
                x = x.to(device, non_blocking=pin_memory)
                y = y.to(device, non_blocking=pin_memory)

                logits = model(x)
                loss = criterion(logits, y)

                if mode == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                bs = x.size(0)
                loss_sum += loss.item() * bs
                predictions = logits.argmax(dim=1)

                correct += (predictions == y).sum().item()
                total += bs

        loss = loss_sum / total
        accuracy = correct / total

        return accuracy, loss


    print(f"Host: {socket.gethostname()}", flush=True)
    print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}", flush=True)

    with utilities.Timer("All epochs:"):
        for epoch in range(start_epoch, epochs):
            train_acc, train_loss = do_epoch("train")
            valid_acc, valid_loss = do_epoch("valid")

            print(
                f"Epoch:{epoch+1}/{epochs}",
                f"Train: Accuracy {train_acc:.3f}; Loss {train_loss:.3f}",
                f"Valid: Accuracy {valid_acc:.3f}; Loss {valid_loss:.3f}",
                flush=True
            )

            save_ckpt(outdir / "last.pt", epoch + 1, model, optimizer, best_valid_acc)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                save_ckpt(outdir / "best.pt", epoch + 1, model, optimizer, best_valid_acc)


if __name__ == "__main__":
    main()
