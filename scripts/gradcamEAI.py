# Author : Mays Zuabi
# Branch : mays/mayss_contribution
# Purpose: Explainable AI using Grad-CAM for emotion recognition models
# Status : In Progress

# Using for now an already existing implementation of Grad-CAM, since this is allowed, to focus more on training the model on our dataset and adapting

import torch
from pytorch_grad_cam import GradCAM

model = ... #placeholder for loading the trained emotion recognition model
model.eval()

target_layers = [model.layer4]  # placeholder for the target layer in the model

def get_gradcam(model, target_layers):
    cam = GradCAM(model=model, target_layers=target_layers)
    return cam