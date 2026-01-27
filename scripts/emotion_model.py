# Author : Mays Zuabi
# Branch : mays/mayss_contribution
# Purpose : Script for defining the model architecture and training
# Status : done - model trained and saved as emotion_model.pt


import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

EMOTION_DICT = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
}

NUM_CLASSES = len(EMOTION_DICT)

# custom Dataset for loading the balanced RAF-DB dataset
class BalancedDataset(Dataset):
    def __init__(self, split):
        self.root = Path(f"data/balanced-raf-db/{split}")
        self.df = pd.read_csv(self.root / "labels.csv")

        self.transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.df) 
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.root / row["filename"]).convert('L') #
        label = int(row["label"]) #label as integer
        
        image = self.transform(image)
        return image, label
     
class ResNetEmotionModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.model = resnet18(weights=None) # for training from scratch
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                     stride=2, padding=3, bias=False)  
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        

    def forward(self, x):
        return self.model(x) 
    
    #training, validation, and testing
    def run_epoch(model, data_loader, criterion, optimizer=None, device="cpu"):
        is_training = optimizer is not None
        model.train() if is_training else model.eval()

        epoch_loss, correct, total = 0.0, 0, 0

        # gradient tracking only if in training mode
        with torch.set_grad_enabled(is_training):
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                if is_training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_loss = epoch_loss / len(data_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy
    
    def training_model(epochs=5, batch_size=64, lr=1e-3):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNetEmotionModel().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
        criterion = nn.CrossEntropyLoss()

        loaders = {
            split: DataLoader(
                BalancedDataset(split),
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=4,
                pin_memory=True
            ) for torch.split in ["train", "val", "test"]
        }

        epoch_losses = []
        epoch_accuracies = []

        for epoch in tqdm(range(epochs), desc='Epoch'):
            train_loss, train_acc = ResNetEmotionModel.run_epoch(
                model, loaders["train"], criterion, optimizer, device
            )
            val_loss, val_acc = ResNetEmotionModel.run_epoch(
                model, loaders["val"], criterion, None, device
            )

            epoch_losses.append(train_loss)
            epoch_accuracies.append(train_acc)

            tqdm.write(f"Epoch [{epoch+1}/{epochs}], \n Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}% \n Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        
        # test evaluation
        test_loss, test_acc = ResNetEmotionModel.run_epoch(
            model, loaders["test"], criterion, None, device
        )
        print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

        #saving the trained model
        torch.save(model.state_dict(), 'emotion_model.pt') 
        print("\nModel saved as emotion_model.pt")
        
        #plotting the loss curve
        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        plt.plot(epoch_losses, marker='o')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        plt.subplot(1,2,2)
        plt.plot(epoch_accuracies, marker='+')
        plt.title('Training Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print(f"Total parameters: {sum([_.numel() for _ in ResNetEmotionModel().parameters()])}")
    ResNetEmotionModel.training_model(epochs=5, batch_size=64, lr=1e-3)