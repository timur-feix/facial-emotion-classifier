# Author : Mays Zuabi
# Branch : mays/mayss_contribution
# Purpose : Script for defining the model architecture and training
# Status : In Progress

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
        self.df = pd.read_csv(self.root / "labels.csv") #
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.df) 
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(self.root / row["filename"]).convert('L') #
        label = EMOTION_DICT.index(row["label"])

        if self.transform:
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
    
    def training_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ResNetEmotionModel().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) #lr placeholder
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(
            BalancedDataset("train"),
            batch_size=64,
            shuffle=True
        )

        losses = []
        correct = 0
        total = 0

        for epoch in tqdm(range(10), desc='Epoch'):
            model.train()
            for step, [example, label] in enumerate(tqdm(train_loader, desc='Batch')): 
                    predictions = model(example)
                    loss = criterion(predictions, label)
                    #clearing old gradients
                    optimizer.zero_grad()
                    #new gradients 
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item()) 

                    correct += (predictions == label).sum().item()
                    total += label.size(0)
            print(f"Epoch {epoch+1} finished, Loss: {loss.item()}"
                f"accuracy: {100. * correct / total}")
        
        #saving the trained model
        torch.save(model.state_dict(), 'emotion_model.pt') 
        return losses
    
    def validate_model(self):
        plt.plot(losses)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.show()
        
print(f"the sum of parameters: {sum([_.numel() for _ in ResNetEmotionModel().parameters()])}")
