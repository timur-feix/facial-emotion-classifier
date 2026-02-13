# Author : Mays Zuabi
# Branch : mays/mayss_contribution
# Purpose: Explainable AI using Grad-CAM for emotion recognition models
# Status : Code Completed - script visualizes Grad-CAM results on test images
# Note : still needs integration with the trained model weights and combining with demos

# Using for now an already existing implementation of Grad-CAM, since this is allowed, to focus more on training the model on our dataset and adapting


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import matplotlib.pyplot as plt

def compute_gradcam(model, image_tensor, class_index, device):
    image_tensor = image_tensor.to(device)

    # LightResNet18 için son conv layer
    target_layers = [model.layer4[-1]]

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(class_index)]

    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    return grayscale_cam[0]



def visualize_results(image_pil, cam, predicted_label):
    cam_h, cam_w = cam.shape
    image_resized = image_pil.resize((cam_w, cam_h))

    img_np = np.array(image_resized).astype(np.float32) / 255.0
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)

    cam_image = show_cam_on_image(img_np, cam, use_rgb=True)

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(img_np)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title(f"Grad-CAM (Predicted: {predicted_label})")
    plt.imshow(cam_image)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
