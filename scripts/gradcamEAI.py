# Author : Mays Zuabi
# Branch : mays/mayss_contribution
# Purpose: Explainable AI using Grad-CAM for emotion recognition models
# Status : In Progress

# Using for now an already existing implementation of Grad-CAM, since this is allowed, to focus more on training the model on our dataset and adapting

from scripts.emotion_model import ResNetEmotionModel, EMOTION_DICT

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
from PIL import Image
from pathlib import Path

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# Load our pre-trained model
def load_model(weights_path='emotion_model.pt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    res_model = ResNetEmotionModel(num_classes=len(EMOTION_DICT)) 
    res_model.load_state_dict(torch.load(weights_path,
                                          map_location=torch.device('cpu')))
    res_model.to(device)
    res_model.eval()

    return res_model, device

# Load the test images
def load_test_data(index = 0):
    test_path = Path("data/balanced-raf-rb/test")  # path to test dataset
    csv_path = test_path / "labels.csv"

    df = pd.read_csv(csv_path)

    row = df.iloc[index]
    image_path = test_path / row["filename"]
    label = EMOTION_DICT.index(row["label"])
    image_pil = Image.open(image_path).convert('L')  # convert to grayscale
    transform = transforms.ToTensor()
    image_tensor = transform(image_pil).unsqueeze(0)  # add batch dimension
    return image_pil, image_tensor, label
    

#prediction pass
def get_prediction(res_model, image, device):
    image = image.to(device)

    with torch.no_grad():
        output = res_model(image)
        predicted = torch.argmax(output, 1).item()
        emotion_name = EMOTION_DICT[predicted]
    print(f'Predicted Emotion: {emotion_name}')
    return predicted 
     
# Grad-CAM setup:actual usage of the existing gradcam
def compute_gradcam(res_model, image_tensor, predicted):
    target_layers = [res_model.model.layer4[-1]] # target layer in the model bzw. last conv layer
    
    cam = GradCAM(model=res_model,
                  target_layers=target_layers)
    targets = [ClassifierOutputTarget(targeted=target_class)]
    grayscale_cam = cam(input_tensor=image_tensor,
                         targets=targets)
    
    return grayscale_cam[0]

def visualize_results(image, cam):
    #overlay gradcam on image
    # Convert tensor to numpy array for visualization
    img_np = np.array(image_pil).astype(np.float32) / 255.0
    img_np = np.stack([img_np.squeeze()] * 3, axis=-1)  # Convert to 3 channels RGB
    # Overlay Grad-CAM on the image
    cam_image = show_cam_on_image(img_np, cam, use_rgb=True)

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.title('Original Image')
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title('Grad-CAM')
    plt.imshow(cam_image)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    res_model, device = load_model()
    image_pil, image_tensor = load_test_data()
    predicted = get_prediction(res_model, image_tensor, device)
    cam = compute_gradcam(res_model, image_tensor.to(device), predicted)
    visualize_results(image_pil, cam)