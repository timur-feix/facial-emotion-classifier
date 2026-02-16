# Branch : mays/mayss_contribution
# Purpose: Demo script for video emotion recognition 
# Input : video file path
# Output : video file with emotion labels and Grad-CAM overlay (if enabled)

import cv2
import torch
import numpy as np
from pathlib import Path
from collections import Counter

from src.dataset import INDEX_MAP as EMOTION_DICT
from scripts.gradcamEAI import load_model, compute_gradcam

class VideoDemo:
    def __init__(self, video_path, model_weights='checkpoints/best.pt',
                   output_path="output_with_gradcam.mp4", enable_gradcam=True,
                   window_seconds=1, confidence_threshold=0.5):
        self.model, self.device = load_model(model_weights)
        self.model.eval()
        self.emotion_dict = {v: k for k, v in EMOTION_DICT.items()}

        self.video_path = str(video_path)
        self.output_path = str(output_path)
        self.enable_gradcam = enable_gradcam
        self.window_seconds = window_seconds
        self.confidence_threshold = confidence_threshold

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    def preprocess_face(self, face):
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (64, 64))
        face_normalized = face_resized.astype("float32") / 255.0

        face_tensor = torch.from_numpy(face_normalized).unsqueeze(0).unsqueeze(0).to(self.device)
        # [1,1,64,64] -> [1,3,64,64]
        face_tensor = face_tensor.repeat(1, 3, 1, 1)

        # if you trained with (x-0.5)/0.5
        face_tensor = (face_tensor - 0.5) / 0.5

        return face_tensor
    
    def run(self):
        cap_video = cv2.VideoCapture(self.video_path)

        if not cap_video.isOpened():
            raise RuntimeError(f"Error: Could not open video source {self.video_path}.")
        
        fps = int(cap_video.get(cv2.CAP_PROP_FPS))
        width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_per_window = fps * self.window_seconds
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        frame_count = 0
        dominant_emotion = "N/A"
        emotion_window = []

        print(f"Processing video: {self.video_path}")
        print(f"Output will be saved to {self.output_path}")

        while True:
            ret, frame = cap_video.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1,
                                                       minNeighbors=5, minSize=(30, 30))
            
            current_emotion = "no person"
            for (x, y, w, h) in faces:
                x1, y1 = max(x, 0), max(y, 0)
                x2, y2 = min(x +w, width), min(y + h, height)
                
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                face_tensor = self.preprocess_face(face)

                with torch.no_grad():
                    outputs = self.model(face_tensor)
                    probs =torch.softmax(outputs, dim=1)
                    predicted = torch.argmax(probs, 1).item()
                    confidence = probs[0, predicted].item()
                    current_emotion = self.emotion_dict[predicted]
                    
                # only update if confidence is above threshold
                if confidence >= self.confidence_threshold:
                    current_emotion = self.emotion_dict[predicted]
                    emotion_window.append(predicted)

                if self.enable_gradcam:
                    cam = compute_gradcam(self.model, face_tensor,
                                              predicted,
                                              self.device)
                    cam_resized = cv2.resize(cam, (x2 - x1, y2 - y1))

                    heatmapped = cv2.applyColorMap(np.uint8(255 * cam_resized),
                                                  cv2.COLORMAP_JET)
                    
                    frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.5,
                                                          heatmapped, 0.5, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 0), 2)
                cv2.putText(frame, f"{current_emotion} ({confidence*100:.1f}%)",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (15, 15, 15), 2)
            
            frame_count += 1
            # Update dominant emotion every window
            if frame_count % frames_per_window == 0 and emotion_window:
                dominant_idx = Counter(emotion_window).most_common(1)[0][0]
                dominant_emotion = self.emotion_dict[dominant_idx]
                emotion_window.clear()

            cv2.putText(frame,
                        f"Dominant emotion: {dominant_emotion}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        (0,0,0),
                        6,)
            cv2.putText(frame,
                        f"Dominant emotion: {dominant_emotion}",
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        (255, 255, 255),
                        2,)

            out_video.write(frame)
            
        cap_video.release()
        out_video.release()
        print("Video processing completed.")

if __name__ == "__main__":
    video_path = Path('videos/video_with6emotions.mp4') 
    output_path = Path('videos/video_with_emotion.mp4')
      
    video_demo = VideoDemo(video_path=video_path,
                           model_weights='checkpoints/best.pt',
                           output_path=output_path,
                           enable_gradcam=True,
                           window_seconds=1)
    video_demo.run()