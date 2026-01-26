# Author : Mays Zuabi
# Branch : mays/mayss_contribution
# Purpose : real-time webcam demo 
# Status : In Progress
import cv2
import time
import torch
import os 
import numpy as np
import mediapipe as mp

from scripts.emotion_model import EMOTION_DICT
from scripts.gradcamEAI import load_model, compute_gradcam
class WebcamDemo:
    def __init__(self, model_weights='emotion_model.pt'):
        self.res_model, self.device = load_model(model_weights)
        self.res_model.eval()
        self.emotion_dict = EMOTION_DICT
        self.gradcam_enabled = False

        # Initialize Mediapipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
                            model_selection=0,
                            min_detection_confidence=0.5)
        
        self.WINDOW_NAME = "Webcam Emotion Recognition Demo"
    
    #prepare face for model input resolution
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
        

# Load the pre-trained emotion recognition model
base_option = Python.BaseOption(model_assest_path='emotion_model.pt')
optipons 
# Grad-CAM & overlay

# frame and showing face box
fps = cap_webcam.get(cv2.CAP_PROP_FPS)
frames_number = 0 

    #webcam implementation with and without gradcam overlay
    def webcam_hybrid(model, device):




# window variables
full_screen = False
cam_height = 480
cam_width = 640
cv2.resizeWindow(WINDOW_NAME, cam_width, cam_height)
cv2.moveWindow(WINDOW_NAME, 0, 0)
cv2.setWindowTitle(WINDOW_NAME, WINDOW_NAME)



# fetch face
for face_results in results:
    ret, frame = cap_webcam.read()
    cv2.imshow('Webcam Emotion Recognition', frame)
    cv2.circle(frame, (50, 50), 10, (0, 255, 0), -1)  
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), 2)

    cv2.waitKey(0) # Waits indefinitely until a key is pressed



for face_results in face_detection.processes: #
    ret, frame = cap_webcam.read()
    if not ret:
        break

    frames_number += 1
    start_time = time.time()

# display predicted emotion on the video feed
detected_result = mp_face_detection.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #


# FPS calculation
    end_time = time.time()
    fps = frames_number / (end_time - start_time)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# closing the webcam
cap_webcam.release()
cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Starting Webcam Emotion Recognition Demo...\n")
    emotion_matcher = WebcamEmotionMatcher()
    emotion_matcher.run()