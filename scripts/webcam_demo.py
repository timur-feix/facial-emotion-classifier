# Author : Mays Zuabi
# Branch : mays/mayss_contribution
# Purpose : real-time webcam demo (Hybrid mode with and without Grad-CAM overlay)
# Status : In Progress
import cv2
import time
import torch 
import numpy as np

from scripts.emotion_model import EMOTION_DICT
from scripts.gradcamEAI import load_model, compute_gradcam

class WebcamDemo:
    
    def __init__(self, model_weights='emotion_model.pt'):
        self.model, self.device = load_model(model_weights)
        self.model.eval()
        self.emotion_dict = EMOTION_DICT
        self.gradcam_enabled = False

        # Initialize Face Recognition
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        self.WINDOW_NAME = "Webcam Emotion Recognition Demo"
    
    # prepare face for model input 
    def preprocess_face(self, frame):
        face_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (64, 64))
        face_normalized = face_resized / 255.0
        face_tensor = torch.tensor(face_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # add batch and channel dims
        return face_tensor.to(self.device)

    def run(self):
        # Open a connection to the webcam
        cap_webcam = cv2.VideoCapture(0) 
        if not cap_webcam.isOpened():
            print("Error: Could not open camera.")
            exit()
        
        start_time = time.time()
        fps = 0.0
        font = cv2.FONT_HERSHEY_SIMPLEX

        print("Controls:")
        print("Press 'g' -> to toggle Grad-CAM overlay.")
        print("Press 'q' -> to quit.")

        running = True
        while running:
            ret, frame = cap_webcam.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

            # Process each detected face (if any)
            if faces is not None:
                for (x, y, w, h) in faces:
                    x1 = max(x, 0)
                    y1 = max(y, 0)
                    x2 = min(x + w, frame.shape[1])
                    y2 = min(y + h, frame.shape[0])

                    # Extract face Region of Interest (ROI)
                    face_roi = frame[y1:y2, x1:x2]
                    # check if face_roi is out of frame
                    if face_roi.size == 0:
                        continue
                    face_tensor = self.preprocess_face(face_roi)

                    # Model prediction
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
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, emotion_label,
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (36,255,12), 2)

            # Calculate and display FPS
            end_time = time.time()
            fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0.0
            start_time = end_time
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

            status = "Grad-CAM ON" if self.gradcam_enabled else "Grad-CAM OFF"
            cv2.putText(frame, status, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            
            # Title bar
            cv2.putText(frame, "Webcam Emotion Recognition Demo",
                        (frame.shape[1]//2 - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 0, 0), 2)
            # Status bar 
            self.draw_text(frame, "Press 'g' to toggle Grad-CAM, 'q' to quit",
                            pos_type=(10, frame.shape[0] - 10))
            

            cv2.imshow(self.WINDOW_NAME, frame)

            # Handle key presses every frame (works even with no faces)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('g'):
                self.gradcam_enabled = not self.gradcam_enabled
            elif key == ord('q'):
                running = False
                break


        # closing the webcam
        cap_webcam.release()
        cv2.destroyAllWindows()

    def draw_text(img, text, font_scale = 0.6, color=(0, 255, 0), bg_color =(0, 0, 0),
                           thickness=2, pos_type='bottom_left', padding=5):
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                cv2.rectangle(img,
                              (10 - padding, img.shape[0] - 10 - text_size[1] - padding),
                              (10 + text_size[0] + padding, img.shape[0] - 10 + padding),
                              bg_color, -1)
                cv2.putText(img, text, pos_type, font, font_scale, color, thickness , cv2.LINE_AA)
            

if __name__ == "__main__":
    print("Starting Webcam Emotion Recognition Demo...\n")
    live_demo = WebcamDemo()
    live_demo.run()