# Author : Neslihan Bir 
# Branch : mays/mayss_contribution
# Purpose: Demo script for video emotion recognition 

from scripts.emotion_model import ResNetEmotionModel, EMOTION_DICT

import cv2
import torch
import numpy as np
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam_utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# Load pre-trained model
def load_model (weigths_path="emotion_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_avaible() else "cpu")

    model = ResNetEmotionModel(num_classes=len(EMOTION_DICT))
    model.load_state_dict(torch.load(weigths_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device

# Preprocessing transform

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Noramalize(mean=[0.5], std=[0.5])
])

# Video Demo

def run_video_demo(video_path):

    model, device = load_model()

    # GradCam setup (same layer as before)
    target_layers = [model.model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpend():
        print("Could not open video.")
        return 
    
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 1. Preprocess frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image_tensor = transform(frame_rgb).unsquezze(0).to(device)

        # 2. Prediction
        with torch.no_grad():
            output = model(image_tensor)
            predicted = torch.argmax(output, 1).item()

        # 3. GradCam
        targets = [ClassifierOutputTarget(predicted)]
        grayscale_cam = cam(input_tensor=image_tensor, targets = targets)[0]

        # 4. Overlay heatmap on frame
        frame.float = frame_rgb.astype(np.float32) / 255.0
        cam_image = show_cam_on_image(frame_float, grayscale_cam, use_rgb=True)

        # 5. Show result
        emotion_text = EMOTION_DICT[predicted]

        cv2.putText(
            cam_image,
            f"Emotion: {emotion_text}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255);
            2
        )

        cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("GradCAM Video Demo", cam_image_bgr)

        #press q to quit 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        
    cap.release()
    cv2.destroyAllWindows()

#Start script
if __name__ == "__main__":
    run_video_demo("test")