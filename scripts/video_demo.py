# Author : Mays Zuabi
# Branch : mays/mayss_contribution
# Purpose: Demo script for video emotion recognition 


#Pipeline
#1) Loads a trained ResNetEmotionModel (.pth)
#2) Reads an input video frame-by-frame
#3) Predicts emotion for each processed frame
#4) Computes Grad-CAM for the predicted class
#5) Overlays heatmap + label on the original frame 
#6) Writes an output video


import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from scripts.emotion_model import ResNetEmotionModel, EMOTION_DICT

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

#--------------
# Model Loading
#--------------
def load_model(weigths_path: str, num_classes: int):
    #Loads your trained ResNetEmotionModel from a pth state dict file 
    #Returns: (model, device)

    device = tourch.device ("cuda" if tourch.cuda.is_avaible() else "cpu")

    model = ResNetEmotionModel(num_classes=num_classes)

    weigths_path = Path(weigths_path)
    if not weigths_path.exists():
        raise FileNotFoundError(f"Model. weigths not found: {weigths_path}")
    
    state = tourch.load(weigths_path, map_location="cpu")
    #Support checkpoints that store state_dict under a key
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    return model, device 

#----------------
#Video Preprocessing 
#-----------------

def preprocess_frame(frame_bgr: np.ndarray,size: int = 64, grayscale: bool = False) -> tuple[torch.Tensor, np.ndaray]:

    #Convert BGR ->RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    #O force grayscale if model was trained on grayscale
    if grayscale: 
     gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)     #[H,W]
     rgb = np.stack([gray, gray, gray], axis=-1)      #[H,W,3]

    #Reesize to model input size
    rgb_resized = cv2.resize(rgb, (size,size), interpolation = cv2.INTER_AREA)

    #For overlay we need float RGB in [0,1]
    rgb01 = rgb_resized.astype(np.float32) / 255.0

    #Normalize for ResNet 
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array ([0.229, 0.224, 0.225], dytype=np.float32)
    norm = (rgb_01 - mean) / std

    #HWC -> CHW, add batch dimension 
    input_tensor = torch.from_numpy(norm).permute(2, 0, 1).unsquezze(0)

    return input_tensor, rgb_01


#Writes readable label on the frame
def put_label (frame_bgr: np.ndarray, text: str) -> np.ndarray:
   x, y = 20,40
   cv2.putText(frame_bgr, text, (x,y),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
   cv2.putText(frame_bgr,text, (x,y),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 5, cv2.LINE_AA)
   return frame_bgr


#-----------------
#GradCam Setup 
#-----------------
def build_cam(model: torch.nn.Module):
   # IMPORTANT: target layer inside wrapper -> model.model.layer4[-1]
    target_layers = [model.model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    return cam


def predict_emotion(model, x: torch.Tensor, device) -> tuple[int, float, np.ndarray]:
    """
    Runs model inference on one image tensor.
    Returns: (pred_class_idx, confidence, probs_array)
    """
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)                   # [1, num_classes]
        probs = F.softmax(logits, dim=1)[0] # [num_classes]
        pred = int(torch.argmax(probs).item())
        conf = float(probs[pred].item())

    return pred, conf, probs.detach().cpu().numpy()


def compute_gradcam_heatmap(cam: GradCAM, x: torch.Tensor, class_idx: int) -> np.ndarray:
    #computes a gradcam heatmap (values 0..1) for a given class

    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor =x, targets=targets)  #shape: [B,H,W]
    heatmap = grayscale_cam[0]
    return heatmap

#---------
#Main Video Demo 
#-----------

def run_video_demo(video_in: str,
                   video_out:str,
                   size:int = 64,
                   every_n: int = 1,
                   grayscale: bool = False,
                   max_frames: int | None = None):
    
    # Load model
    model, device = load_model(weights_path, num_classes=len(EMOTION_DICT))
    cam = build_cam(model)

    # Open input video
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_out, fourcc, fps, (W, H))
    if not out.isOpened():
        raise RuntimeError(f"Could not open output video for writing: {video_out}")

    print("----- Video Demo Settings -----")
    print(f"Input:  {video_in}")
    print(f"Output: {video_out}")
    print(f"FPS: {fps:.2f}, Resolution: {W}x{H}")
    print(f"Model: {weights_path}")
    print(f"Process every Nth frame: {every_n}")
    print(f"Grayscale preprocess: {grayscale}")
    print("--------------------------------")

    idx = 0
    processed = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if max_frames is not None and processed >= max_frames:
            break

        # If not processing this frame, write as-is
        if idx % every_n != 0:
            out.write(frame_bgr)
            idx += 1
            continue

        # Preprocess
        x, rgb_01 = preprocess_frame(frame_bgr, size=size, grayscale=grayscale)

        # Predict
        pred_idx, conf, _ = predict_emotion(model, x, device)
        emotion_name = EMOTION_DICT[pred_idx]
        label_text = f"{emotion_name} ({conf:.2f})"

        # Grad-CAM heatmap
        heatmap = compute_gradcam_heatmap(cam, x.to(device), pred_idx)

        # Overlay heatmap onto the resized rgb (size x size)
        cam_rgb = show_cam_on_image(rgb_01, heatmap, use_rgb=True)  # returns uint8 RGB image

        # Resize overlay back to original video resolution
        cam_bgr = cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR)
        cam_bgr = cv2.resize(cam_bgr, (W, H), interpolation=cv2.INTER_LINEAR)

        # Add label
        cam_bgr = put_label(cam_bgr, label_text)

        # Write output
        out.write(cam_bgr)

        idx += 1
        processed += 1

        if processed % 30 == 0:
            print(f"Processed frames: {processed}")

    cap.release()
    out.release()
    print(f"Done. Saved output video to: {video_out}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Emotion Recognition + Grad-CAM Demo")
    parser.add_argument("--video_in", type=str, required=True, help="Path to input video (e.g. input.mp4)")
    parser.add_argument("--video_out", type=str, required=True, help="Path to output video (e.g. out.mp4)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained model weights (.pt/.pth)")
    parser.add_argument("--size", type=int, default=64, help="Model input size (default: 64)")
    parser.add_argument("--every_n", type=int, default=1, help="Process every Nth frame (default: 1)")
    parser.add_argument("--grayscale", action="store_true",
                        help="Convert frames to grayscale and stack to 3 channels (if model trained on grayscale)")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional: stop after N processed frames")

    args = parser.parse_args()

    run_video_demo(
        video_in=args.video_in,
        video_out=args.video_out,
        weights_path=args.ckpt,
        size=args.size,
        every_n=args.every_n,
        grayscale=args.grayscale,
        max_frames=args.max_frames
    )



#So video im terminal starten 
#python video_demo.py \
# --video_in test.mp4 \
# --video_out result.mp4 \
# --ckpt emotion_model.pt

#In Terminal NewTerminal VSCode
#python3 video_demo.py --video_in meineaufnahme.mov --video_out out.mp4 --ckpt emotion_model.pt

