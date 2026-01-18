# Author : Mays Zuabi
# Branch : mays/mayss_contribution
# Purpose: Explainable AI using Grad-CAM for emotion recognition models
# Status : In Progress

# Using for now an already existing implementation of Grad-CAM, since this is allowed, to focus more on training the model on our dataset and adapting

from scripts.emotion_model import ResNetEmotionModel, EMOTION_DICT
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
import matplotlib.pyplot as plt

# Load our pre-trained model
res_model = ResNetEmotionModel() 
#res_model.load_state_dict(torch.load('models/resnet_emotion_model.pth', map_location=torch.device('cpu')))
res_model.eval()

test_path = 'data/test'  # path to test dataset
test_dataset = res_model.get_dataloader(test_path, batch_size=1).dataset
num_classes = len(EMOTION_DICT)

# getting samples from the dataset
image, target = test_dataset.__getitem__(4)  # assuming test_dataset is defined elsewhere

target = torch.argmax(target).item()  # Convert one-hot to class index if necessary
target_layers = [res_model.model.layer4]  # placeholder for the target layer in the model

#prediction pass

def get_prediction(model, input_tensor):
    pass

#def get_edge_detection(input_image):
edges = cv2.Canny(image, 100, 200)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Original Image')
plt.imshow(image.permute(1, 2, 0).cpu().numpy())
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Edge Detection')
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.show()



def get_gradcam(model, target_layers):
    cam = GradCAM(model=res_model, target_layers=target_layers)
    return cam