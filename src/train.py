import dataset
from torch.utils.data import DataLoader

import utilities

from model import FacialEmotionRecognitionCNN as FERCNN

from sys import argv
import torch

using_debug = "--debug" in argv

def main():
    # timing supervisors
    optimizer_timing_supervisor = [0.0, 0.0]
    backwards_timing_supervisor = [0.0, 0.0]

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if using_debug: print(f"Using device: {device}")

    model = FERCNN().to(device)

    # loss
    criterion = torch.nn.CrossEntropyLoss()

    #optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    if using_debug: print(f"Using optim: {optimizer}")

    # maybe later: LR schedule depending on whether we want SGD
    # TODO


    # datasets
    train_ds = dataset.RAFDataset("./data/balanced-raf-db/train")
    train_ld = DataLoader(train_ds,
                        batch_size=16,
                        shuffle=True,
                        pin_memory=(device.type == "cuda"),
                        num_workers=0,
                        persistent_workers=False,
                        drop_last=True)

    valid_ds = dataset.RAFDataset("./data/balanced-raf-db/val")
    valid_ld = DataLoader(valid_ds,
                        batch_size=16,
                        shuffle=False,
                        pin_memory=(device.type == "cuda"),
                        num_workers=0,
                        persistent_workers=False,
                        drop_last=False)

    # epoch loop
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
            else: return 0, 0
        
        loss_sum, correct, total = 0, 0, 0
        with context_manager:
            for x, y in data_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)

                if mode == "train":
                    optimizer.zero_grad()
                    with utilities.Timer("loss backward", show=False) as bwt:
                        loss.backward()
                    backwards_timing_supervisor[0] += bwt.timer
                    backwards_timing_supervisor[1] += 1

                    with utilities.Timer("optimizer step", show=False) as opt:
                        optimizer.step()
                    optimizer_timing_supervisor[0] += opt.timer
                    optimizer_timing_supervisor[1] += 1
                    

                bs = x.size(0)
                loss_sum += loss.item() * bs
                predicitions = logits.argmax(dim=1)

                correct += (predicitions == y).sum().item()
                total += bs
            
        loss = loss_sum / total
        accuracy = correct / total

        return accuracy, loss

        
    epochs = 3
    for epoch in range(epochs):
        model.reset_timing_supervisor()
        train_ds.reset_timing_supervisor()

        optimizer_timing_supervisor = [0.0, 0.0]
        backwards_timing_supervisor = [0.0, 0.0]

        with utilities.Timer("epoch total"):
            train_acc, train_loss = do_epoch("train")
            valid_acc, valid_loss = do_epoch("valid")

        print(f"Epoch:{epoch+1}/{epochs}",
            f"Train: Accuracy {train_acc:.3f}; Loss {train_loss:.3f}",
            f"Valid: Accuracy {valid_acc:.3f}; Loss {valid_loss:.3f}")
        
        print("\n")
        print(f"average model forward time: {model.timing_supervisor[0] / model.timing_supervisor[1]}s")
        print(f"total model forward time: {model.timing_supervisor[0]}s")

        print("\n")
        print(f"average train __getitem__ time: {train_ds.timing_supervisor[0] / train_ds.timing_supervisor[1]}s")
        print(f"total train __getitem__ time: {train_ds.timing_supervisor[0]}s")

        print("\n")
        print(f"average optimizer time: {optimizer_timing_supervisor[0] / optimizer_timing_supervisor[1]}s")
        print(f"total optimizer time: {optimizer_timing_supervisor[0]}s")

        print("\n")
        print(f"average loss backward time: {backwards_timing_supervisor[0] / backwards_timing_supervisor[1]}s")
        print(f"total loss backward time: {backwards_timing_supervisor[0]}s")


if __name__ == "__main__":
    main()