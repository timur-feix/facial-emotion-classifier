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
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

# Load our pre-trained model
def load_model():
    num_classes = len(EMOTION_DICT)
    res_model = ResNetEmotionModel(num_classes=num_classes) 
    res_model.load_state_dict(torch.load('models/resnet_emotion_model.pth',
                                          map_location=torch.device('cpu')))
    res_model.eval()
    return res_model

# Load the test images
def load_test_data(res_model):
    test_path = 'data/test'  # path to test dataset
    test_dataset = res_model.get_dataloader(test_path, batch_size=1).dataset
    # getting samples from the dataset
    test_dataset = load_test_data(res_model)
    image, target = test_dataset[4]  # assuming test_dataset is defined as above
    image = image.unsqueeze(0)  # batch dimension
    return test_dataset, image, target

#prediction pass
def get_prediction(res_model, image):
    with torch.no_grad():
        output = res_model(image)
        predicted = torch.argmax(output, 1).item()
        emotion_name = EMOTION_DICT[predicted]
        print(f'Predicted Emotion: {emotion_name}')
        return predicted      

def get_edge_detection(image):
    # Convert tensor to numpy array for cv2
    img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Ensure it's in the right format (0-255)
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    # Convert to grayscale if needed
    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(img_np, 100, 200)
    return edges

# Grad-CAM setup:actual usage of the existing gradcam
def get_gradcam(res_model, target_layers, predicted):
    target_layers = [res_model.model.layer4[-1]] # placeholder for the target layer in the model bzw. last conv layer
    targets = [ClassifierOutputTarget(predicted)]  # using the predicted class as target

    cam = GradCAM(model=res_model,
                  target_layers=target_layers)
    #heatmap = cam(input_tensor=image.unsqueeze(0), targets=targets, num_classes=num_classes) 

    return cam[0]

def visualize_results(image, cam, edges):
    #overlay gradcam on image
    # Convert tensor to numpy array for visualization
    img_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    img_np = img_np.astype(np.float32) / 255.0

    # Overlay Grad-CAM on the image
    cam_image = show_cam_on_image(img_np, cam, use_rgb=True)

    plt.figure(figsize=(10,5))

    plt.subplot(1,3,1)
    plt.title('Original Image')
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title('Grad-CAM')
    plt.imshow(cam_image)
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title('Edge Detection')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')
    plt.show()