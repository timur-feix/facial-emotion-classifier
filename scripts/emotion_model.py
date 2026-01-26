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
        label = EMOTION_DICT[row["label"]]
        
        image = self.transform(image)
        return image, label
     
class ResNetEmotionModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                     stride=2, padding=3, bias=False)  
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        

    def forward(self, x):
        return self.model(x) 
    
    def training_model(epochs=5, batch_size=64, lr=1e-3):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNetEmotionModel().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(
            BalancedDataset("train"),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        epoch_losses = []
        epoch_accuracies = []

        for epoch in tqdm(range(epochs), desc='Epoch', position=0):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            batch_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}',
                              position=1, leave=False)
            
            for images, label in batch_bar:
                    images = images.to(device)
                    label = label.to(device)
                    #clearing old gradients
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, label)

                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)

                    correct += (predicted == label).sum().item()
                    total += label.size(0)
                    batch_bar.set_postfix(loss=loss.item(),
                                          accuracy=f"{100. * correct / total:.2f}%")
            avg_loss = running_loss / len(train_loader)
            accuracy = 100. * correct / total
            epoch_losses.append(avg_loss)
            epoch_accuracies.append(accuracy)

            tqdm.write(f"Epoch [{epoch+1}/{epochs}], \n Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
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
    print(f"the sum of parameters: {sum([_.numel() for _ in ResNetEmotionModel().parameters()])}")
    ResNetEmotionModel.training_model(epochs=5, batch_size=64, lr=1e-3)