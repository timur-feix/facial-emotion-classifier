# Author : Mays Zuabi
# Branch : mays/mayss_contribution
# Purpose : real-time webcam demo (Hybrid mode with and without Grad-CAM overlay)
# Status : In Progress
import cv2
import time
import torch 
import numpy as np
import mediapipe as mp

from scripts.emotion_model import EMOTION_DICT
from scripts.gradcamEAI import load_model, compute_gradcam
from mediapipe.solutions import face_detection
class WebcamDemo:
    
    def __init__(self, model_weights='emotion_model.pt'):
        self.model, self.device = load_model(model_weights)
        self.model.eval()
        self.emotion_dict = EMOTION_DICT
        self.gradcam_enabled = False

        # Initialize Mediapipe Face Detection
        self.mp_face_detection = mp.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
                            model_selection=0,
                            min_detection_confidence=0.5)
        
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

        print("Controls:")
        print("Press 'g' -> to toggle Grad-CAM overlay.")
        print("Press 'q' -> to quit.")

        while True:
            ret, frame = cap_webcam.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # Convert the frame to RGB for Mediapipe
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(rgb_frame)

            # Process each detected face
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x1 = max(int(bboxC.xmin * w), 0)
                    y1 = max(int(bboxC.ymin * h), 0)
                    x2 = min(int((bboxC.xmin + bboxC.width) * w), w)
                    y2 = min(int((bboxC.ymin + bboxC.height) * h), h)

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
                        cam_image = compute_gradcam(self.model,face_tensor,
                                                     predicted.item(), self.device)
                        cam_image_resized = cv2.resize(cam_image, (x2 - x1, y2 - y1))

                        heatmapped_face = cv2.applyColorMap(np.uint8(255 * cam_image_resized),
                                                             cv2.COLORMAP_JET)
                        frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.5,
                                                               heatmapped_face, 0.5, 0)

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2  ), (0, 255, 0), 2)
                    cv2.putText(frame, emotion_label,
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (36,255,12), 2)
                    
                    # Calculate and display FPS
                    end_time = time.time()
                    fps = 1 / (end_time - start_time)
                    start_time = end_time
                    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)
                    
                    status = "Grad-CAM ON" if self.gradcam_enabled else "Grad-CAM OFF"
                    cv2.putText(frame, status, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2)
                    cv2.imshow(self.WINDOW_NAME, frame) 

                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('g'):
                        self.gradcam_enabled = not self.gradcam_enabled
                    elif key == ord('q'):
                        break


        # closing the webcam
        cap_webcam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting Webcam Emotion Recognition Demo...\n")
    live_demo = WebcamDemo()
    live_demo.run()