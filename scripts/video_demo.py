# Author : Neslihan Bir
# Branch : neslis_contribuition
# Purpose: Demo script for video emotion recognition 
# Status : In Progress (Mays + Neslihan)


import cv2
import torch
import numpy as np
from torchvision import transforms

from src.model import FacialEmotionRecognitionCNN
from pytorch_grad_cam import GradCAM 
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

EMOTION_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
}

def load_model(weights_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        
    model = FacialEmotionRecognitionCNN(n_classes=6)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    model.to(device)
    model.eval()
    return model, device


def compute_gradcam(model, image_tensor, class_index):
    #letzter conv layer im CNN
    target_layers = [model.stacks[-1].conv]

    cam = GradCAM(model= model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(class_index)]
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
    return grayscale_cam[0]

                          
# Preprocessing transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def run_video_demo(video_path: str, weights_path: str = "emotion_model.pt"):
    # 1) Load Model + device
    model, device = load_model(weights_path)

    # 2) Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open Video: {video_path}")
        return

    print("Video opened. Press 'q' to quit.")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # 3) Convert frame (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 4) Preprocess -> tensor [1,3,64,64]
        image_tensor = transform(frame_rgb).unsqueeze(0).to(device)

        # 5) Predict
        with torch.no_grad():
            output = model(image_tensor)
            predicted = torch.argmax(output, 1).item()

        emotion_name = EMOTION_DICT[predicted]

        # 6) GradCAM heatmap (0..1)
        heatmap = compute_gradcam(model, image_tensor, predicted)

        heatmap = cv2.resize(heatmap, (frame_rgb.shape[1], frame_rgb.shape[0]))

        # 7) Overlay on original frame
        rgb_float = frame_rgb.astype(np.float32) / 255.0
        cam_image_rgb = show_cam_on_image(rgb_float, heatmap, use_rgb=True)
        cam_image_bgr = cv2.cvtColor(cam_image_rgb, cv2.COLOR_RGB2BGR)

        # 8) Label
        cv2.putText(
            cam_image_bgr,
            f"Emotion: {emotion_name}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # 9) Show
        cv2.imshow("Video Demo (Emotion + GradCAM)", cam_image_bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    run_video_demo("video.mov", weights_path="checkpoints/best.pt")
