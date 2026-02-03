# Branch : mays/mayss_contribution
# Purpose: Demo script for video emotion recognition 
# Input : video file path
# Output : video file with emotion labels and Grad-CAM overlay (if enabled)

import cv2
import torch
import numpy as np
from pathlib import Path

from scripts.emotion_model import ResNetEmotionModel, EMOTION_DICT
from scripts.gradcamEAI import load_model, compute_gradcam

class VideoDemo:
    def __init__(self, video_path, model_weights='emotion_model.pt',
                   output_path="output_with_gradcam.mp4", enable_gradcam=True):
        self.model, self.device = load_model(model_weights)
        self.model.eval()
        self.emotion_dict = EMOTION_DICT

        self.video_path = str(video_path)
        self.output_path = str(output_path)
        self.enable_gradcam = enable_gradcam

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    def preprocess_face(self, face_img):
        face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (64, 64))
        face_normalized = face_resized / 255.0
        face_tensor = (torch.tensor(face_normalized, dtype=torch.float32)
                       .unsqueeze(0).unsqueeze(0).to(self.device))  
        return face_tensor
    
    def run(self):
        cap_video = cv2.VideoCapture(self.video_path)

        if not cap_video.isOpened():
            raise RuntimeError(f"Error: Could not open video source {self.video_path}.")
        
        fps = cap_video.get(cv2.CAP_PROP_FPS)
        width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        print(f"Processing video: {self.video_path}")
        print(f"Output will be saved to {self.output_path}")

        while True:
            ret, frame = cap_video.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1,
                                                       minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                x1, y1 = max(x, 0), max(y, 0)
                x2, y2 = min(x +w, width), min(y + h, height)
                
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                face_tensor = self.preprocess_face(face_img)

                with torch.no_grad():
                    outputs = self.model(face_tensor)
                    predicted = torch.argmax(outputs, 1).item()
                    emotion_label = self.emotion_dict[predicted]

                if self.enable_gradcam:
                    cam = compute_gradcam(self.model, face_tensor,
                                              predicted,
                                              self.device)
                    cam_resized = cv2.resize(cam, (x2 - x1, y2 - y1))

                    heatmapped = cv2.applyColorMap(np.uint8(255 * cam_resized),
                                                  cv2.COLORMAP_JET)
                    
                    frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.5,
                                                          heatmapped, 0.5, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (44, 150, 104), 2)
                cv2.putText(frame, emotion_label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            out_video.write(frame)
            
        cap_video.release()
        out_video.release()
        cv2.destroyAllWindows()
        print("Video processing completed.")

if __name__ == "__main__":
    video_path = Path('video.mov')
    output_path = Path('video_with_emotion_gradcam.mp4')
      
    video_demo = VideoDemo(video_path=video_path,
                           model_weights='emotion_model.pt',
                           output_path=output_path,
                           enable_gradcam=True)
    video_demo.run()