# src/train.py

import os
from pathlib import Path

import torch
import torch.nn as nn

from dataset import BalancedRafDbDataset
from data_loader import DataLoader
from model import TinyCNN


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    total_seen = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        total_seen += bs

    return total_loss / total_seen, total_acc / total_seen


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    total_seen = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        total_seen += bs

    return total_loss / total_seen, total_acc / total_seen


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    root = "data/balanced-raf-db"
    train_ds = BalancedRafDbDataset(f"{root}/train")
    val_ds = BalancedRafDbDataset(f"{root}/val")
    test_ds = BalancedRafDbDataset(f"{root}/test")

    train_loader = DataLoader(train_ds, batch_size=128, shuffle_=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle_=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle_=False)

    model = TinyCNN(num_classes=6).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x0, y0 = next(iter(train_loader))
    out0 = model(x0.to(device))

    epochs = 10
    best_val_acc = -1.0

    Path("artifacts").mkdir(exist_ok=True)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), "artifacts/tinycnn_best.pth")

    print("Best val acc:", best_val_acc)

    model.load_state_dict(torch.load("artifacts/tinycnn_best.pth", map_location=device))
    te_loss, te_acc = evaluate(model, test_loader, device)
    print(f"TEST | loss {te_loss:.4f} acc {te_acc:.3f}")


if __name__ == "__main__":
    main()
