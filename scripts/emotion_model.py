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
import matplotlib.pyplot as plt

EMOTION_DICT = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
}

class ResNetEmotionModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        

    def forward(self, x):
        return self.model(x) #
    
    def training_model(): #
        optimizer = torch.optim.Adam(ResNetEmotionModel().parameters(), lr=1e-3) #lr placeholder
        criterion = nn.CrossEntropyLoss()
        losses = []
        for epoch in tqdm(range(10), desc='Epoch'):
            ResNetEmotionModel().train()
            for step, [example, label] in enumerate(tqdm(Dl, desc='Batch')): #Dl is placeholder for dataloader
                    predictions = ResNetEmotionModel()(example)
                    loss = criterion(predictions, label)
                    #clearing old gradients
                    optimizer.zero_grad()
                    #new gradients 
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())  #
        ResNetEmotionModel().eval()
        
        #saving the trained model
        torch.save(ResNetEmotionModel().state_dict(), 'emotion_model.pt') 
        return losses
        plt.plot(losses)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.show()
        # Placeholder for continuing validation 
        pass

print(f"the sum of parameters: {sum([_.numel() for _ in ResNetEmotionModel().parameters()])}")
