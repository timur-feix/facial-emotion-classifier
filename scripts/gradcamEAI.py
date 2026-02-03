# Author : Mays Zuabi
# Branch : mays/mayss_contribution
# Purpose: Explainable AI using Grad-CAM for emotion recognition models
# Status : Code Completed - script visualizes Grad-CAM results on test images
# Note : still needs integration with the trained model weights and combining with demos

# Using for now an already existing implementation of Grad-CAM, since this is allowed, to focus more on training the model on our dataset and adapting

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.model import FacialEmotionRecognitionCNN as FERCNN
from src.dataset import INDEX_MAP as EMOTION_DICT


# Load our pre-trained model
def load_model(weights_path='checkpoints/best.pt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = FERCNN(n_classes=len(EMOTION_DICT))

    ckpt = torch.load(weights_path, map_location=device)

    # handle both "full checkpoint" and "pure state_dict" files
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # if trained with DataParallel, strip "module." prefix
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model, device

EMOTION_TO_IDX = {v: k for k, v in EMOTION_DICT.items()}

# Load the test images
def load_test_data():
    test_path = Path("data/balanced-raf-db/test")
    csv_path = test_path / "labels.csv"

    df = pd.read_csv(csv_path)
    row = df.iloc[random.randrange(len(df))]

    image_path = test_path / row["filename"]
    try:
        true_label = EMOTION_DICT[row["label"]]
    except KeyError:
        print(f"EMOTION_DICT:{EMOTION_DICT}\nEMOTION_TO_IDX:{EMOTION_TO_IDX}")
    
    image_pil = Image.open(image_path).convert('L').convert('RGB')  # convert to grayscale
    
    transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]),
    ])

    image_tensor = transform(image_pil).unsqueeze(0)  # type: ignore
    return image_pil, image_tensor, true_label # type: ignore
    

#prediction pass
def get_prediction(model, image_tensor, device):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        predicted = torch.argmax(output, 1).item()

    return predicted 
     
# Grad-CAM setup:actual usage of the existing gradcam
def compute_gradcam(model, image_tensor, class_index, device):
    # ensuring the tensor is on the correct device
    image_tensor = image_tensor.to(device)
    target_layers = [model.stacks[6].conv] # target layer in the model bzw. last conv layer
    
    cam = GradCAM(model=model,
                  target_layers=target_layers)
    targets = [ClassifierOutputTarget(class_index)]
    grayscale_cam = cam(input_tensor=image_tensor,
                         targets=targets) # type: ignore
    
    return grayscale_cam[0]

def visualize_results(image_pil, cam, true_label, predicted):
    #overlay gradcam on image
    # Convert PIL image to match cam size and to numpy for visualization
    cam_h, cam_w = cam.shape
    image_resized = image_pil.resize((cam_w, cam_h))
    img_np = np.array(image_resized).astype(np.float32) / 255.0
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)  # Convert to RGB if grayscale
    # Overlay Grad-CAM on the image (both now share the same spatial size)
    cam_image = show_cam_on_image(img_np, cam, use_rgb=True)

    plt.figure(figsize=(10,5))

    plt.subplot(1,3,1)
    plt.title(f'Original Image - True: {EMOTION_TO_IDX[true_label]}')
    plt.imshow(img_np)
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title(f'Grad-CAM - Predicted: {EMOTION_TO_IDX[predicted]}')
    plt.imshow(cam_image)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Heatmap Only')
    plt.imshow(cam, cmap='jet')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model, device = load_model()
    image_pil, image_tensor, true_label = load_test_data()
    predicted = get_prediction(model, image_tensor, device)
    cam = compute_gradcam(model, image_tensor, predicted, device)
    visualize_results(image_pil, cam, true_label, predicted)