import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from scripts.emotion_model import ResNetEmotionModel, BalancedDataset, EMOTION_DICT #update

def plot_confusion_matrix(model, dataset, device='cpu', classes=None):
  
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    if classes is None:
        classes = [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGn',xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNetEmotionModel() #update
    model.load_state_dict(torch.load("emotion_model.pt", map_location=device))

    test_dataset = BalancedDataset(split="test") #update 

    plot_confusion_matrix(model, test_dataset, device=device, classes=list(EMOTION_DICT.values()))