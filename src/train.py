import dataset, utilities, torch, os, argparse, json, socket, math

from model import FacialEmotionRecognitionCNN as FERCNN
from torch.utils.data import DataLoader

from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--outdir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--debug", action="store_true")

    # LR scheduling
    p.add_argument("--sched", type=str, default="cosine", choices=["none", "cosine", "onecycle"])
    p.add_argument("--warmup", type=float, default=0.05, help="warmup fraction of total steps (cosine)")
    p.add_argument("--min-lr", type=float, default=2e-5, help="eta_min for cosine (absolute LR)")
    return p.parse_args()


def save_ckpt(path, epoch, model, optimizer, best_valid_acc, scheduler=None, global_step=None):
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_valid_acc": best_valid_acc,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if global_step is not None:
        payload["global_step"] = global_step
    torch.save(payload, path)


def load_ckpt(path, model, optimizer, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception as e:
            print(f"[warn] Could not load scheduler state: {e}", flush=True)

    start_epoch = ckpt.get("epoch", 0)
    best_valid_acc = ckpt.get("best_valid_acc", 0.0)
    global_step = ckpt.get("global_step", 0)
    return start_epoch, best_valid_acc, global_step


def main():
    args = parse_args()
    using_debug = args.debug

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

    model = FERCNN().to(device)

    pin_memory = (device.type == "cuda")
    persistent_workers = (num_workers > 0)

    # loss
    criterion = torch.nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-2,  # changed from 1e-4 (helps with late-epoch oscillation/overfit)
    )

    # datasets / loaders
    dl_common = dict(
        batch_size=batch_size,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    train_ds = dataset.RAFDataset(str(data_dir / "train"))
    valid_ds = dataset.RAFDataset(str(data_dir / "val"))

    if num_workers > 0:
        dl_common["prefetch_factor"] = 2

    train_ld = DataLoader(train_ds, shuffle=True, drop_last=True, **dl_common)
    valid_ld = DataLoader(valid_ds, shuffle=False, drop_last=False, **dl_common)

    # scheduler (needs train_ld length)
    scheduler = None
    steps_per_epoch = len(train_ld)
    total_steps = epochs * steps_per_epoch

    if args.sched == "cosine":
        warmup_steps = int(args.warmup * total_steps)
        lr_max = lr
        lr_min = args.min_lr

        def lr_lambda(step: int):
            # warmup
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps

            # cosine decay
            denom = max(1, total_steps - warmup_steps)
            progress = (step - warmup_steps) / denom
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

            # scale factor relative to lr_max
            return (lr_min / lr_max) + (1.0 - (lr_min / lr_max)) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif args.sched == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.2,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )

    # resume
    best_valid_acc = 0.0
    start_epoch = 0
    global_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = root / resume_path
        start_epoch, best_valid_acc, global_step = load_ckpt(resume_path, model, optimizer, scheduler)

    # epoch loop helpers
    def do_epoch(mode: str):
        nonlocal global_step

        if mode == "train":
            context_manager = utilities.NullContext()
            model.train()
            data_loader = train_ld
        elif mode == "valid":
            context_manager = torch.no_grad()
            model.eval()
            data_loader = valid_ld
        else:
            return 0.0, 0.0

        loss_sum, correct, total = 0.0, 0, 0

        with context_manager:
            for x, y in data_loader:
                x = x.to(device, non_blocking=pin_memory)
                y = y.to(device, non_blocking=pin_memory)

                logits = model(x)
                loss = criterion(logits, y)

                if mode == "train":
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    # optional stabilizer (uncomment if you see spiky loss)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    if scheduler is not None:
                        # per-step schedule
                        scheduler.step()
                    global_step += 1

                bs = x.size(0)
                loss_sum += loss.item() * bs
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += bs

        return (correct / total), (loss_sum / total)

    print(f"Host: {socket.gethostname()}", flush=True)
    print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}", flush=True)
    print(f"Scheduler: {args.sched}", flush=True)

    ctx_mgr = utilities.Timer("All epochs") if using_debug else utilities.NullContext()

    with ctx_mgr:
        for epoch in range(start_epoch, epochs):
            train_acc, train_loss = do_epoch("train")
            valid_acc, valid_loss = do_epoch("valid")

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch:{epoch+1}/{epochs} "
                f"Train: Accuracy {train_acc:.3f}; Loss {train_loss:.3f} "
                f"Valid: Accuracy {valid_acc:.3f}; Loss {valid_loss:.3f} "
                f"LR: {current_lr:.2e}",
                flush=True,
            )

            save_ckpt(outdir / "last.pt", epoch + 1, model, optimizer, best_valid_acc, scheduler, global_step)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                save_ckpt(outdir / "best.pt", epoch + 1, model, optimizer, best_valid_acc, scheduler, global_step)


if __name__ == "__main__":
    main()
