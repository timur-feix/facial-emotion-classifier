# Author : Mays Zuabi
# Branch : mays/mayss_contribution
# Purpose : real-time webcam demo (Hybrid mode with and without Grad-CAM overlay)
# Status : In Progress
import cv2
import time
import torch 
import numpy as np

from src.dataset import INDEX_MAP as EMOTION_DICT
from scripts.gradcamEAI import load_model, compute_gradcam

class WebcamDemo:
    def __init__(self, model_weights='checkpoints/best.pt'):
        self.model, self.device = load_model(model_weights)
        self.model.eval()
        self.emotion_dict = {v: k for k, v in EMOTION_DICT.items()}
        self.gradcam_enabled = False

        # Initialize Face Recognition
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        self.WINDOW_NAME = "Webcam Emotion Recognition Demo"
        self.start_time = time.time()
    
    # prepare face for model input 
    def preprocess_face(self, frame):
        # OpenCV gives BGR → convert to RGB
        face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (64, 64))

        # normalize to [0,1]
        face_normalized = face_resized.astype("float32") / 255.0

        # HWC → CHW → NCHW
        face_tensor = (
            torch.from_numpy(face_normalized)
            .permute(2, 0, 1)   # [3,64,64]
            .unsqueeze(0)       # [1,3,64,64]
            .to(self.device)
        )

        # match training normalization
        face_tensor = (face_tensor - 0.5) / 0.5

        return face_tensor
    
    # UI helper 
    def draw_text(self, img, text, x, y, bg_color):
        font = cv2.FONT_HERSHEY_SIMPLEX 
        font_scale = 0.6
        thickness = 2
        padding = 5
        color = (255, 255, 255)
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        cv2.rectangle(img,
                      (x - padding, y - h - padding),
                      (x + w + padding, y + padding),
                      bg_color, -1,)
        cv2.putText(img, text, (x, y), font, font_scale, color,
                     thickness , cv2.LINE_AA)

    def run(self):
        # Open a connection to the webcam
        cap_webcam = cv2.VideoCapture(0) 
        if not cap_webcam.isOpened():
            print("Error: Could not open camera.")
            exit()
        
        running = True
        while running:
            ret, frame = cap_webcam.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame,
                                                        scaleFactor=1.1, minNeighbors=5)

            # Process each detected face (if any)
            if faces is not None:
                for (x, y, w, h) in faces:
                    x1, y1 = max(x, 0), max(y, 0)
                    x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])

                    # Extract face Region of Interest (ROI)
                    face_roi = frame[y1:y2, x1:x2]
                    # check if face_roi is out of frame
                    if face_roi.size == 0:
                        continue
                    face_tensor = self.preprocess_face(face_roi)

                    with torch.no_grad():
                        outputs = self.model(face_tensor)
                        _, predicted = torch.max(outputs, 1)
                        emotion_label = self.emotion_dict[predicted.item()]

                    # Grad-CAM overlay
                    if self.gradcam_enabled:
                        cam_image = compute_gradcam(self.model, face_tensor,
                                                     predicted.item(), self.device)
                        cam_image_resized = cv2.resize(cam_image, (x2 - x1, y2 - y1))

                        heatmapped_face = cv2.applyColorMap(np.uint8(255 * cam_image_resized),
                                                             cv2.COLORMAP_JET)
                        frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.5,
                                                               heatmapped_face, 0.5, 0)

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (44, 150, 104), 6)
                    self.draw_text(frame, emotion_label,
                                x1, y1-20,
                                bg_color =(0, 0, 0),)

            # Calculate and display FPS
            current_time = time.time()
            fps = 1 / (current_time - self.start_time) if (current_time - self.start_time) > 0 else 0.0
            self.start_time = current_time

            # Title bar
            title = "Webcam Emotion Recognition Demo"
            (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 10)
            self.draw_text(frame, title,
                        frame.shape[1] // 2 - tw // 2,
                        30,
                        bg_color =(10, 10, 10),)

            self.draw_text(frame, f'FPS: {fps:.2f}', 10, 30,
                        bg_color =(50, 50, 50),)

            status = "Grad-CAM ON" if self.gradcam_enabled else "Grad-CAM OFF"
            self.draw_text(frame, f"[G] toggle Grad-CAM | [Q] Quit | {status}",
                        10, 65,
                        bg_color =(50, 50, 50),)
            
            
            cv2.imshow(self.WINDOW_NAME, frame)

            # Handle key presses every frame (works even with no faces)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('g') or key == ord('G'):
                self.gradcam_enabled = not self.gradcam_enabled
            elif key == ord('q') or key == ord('Q'):
                running = False
                break


        # closing the webcam
        cap_webcam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting Webcam Emotion Recognition Demo...\n")
    live_demo = WebcamDemo()
    live_demo.run()

    print("Controls:")
    print("Press 'g' -> to toggle Grad-CAM overlay.")
    print("Press 'q' -> to quit.")