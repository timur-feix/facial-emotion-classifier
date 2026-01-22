# Author : Mays Zuabi
# Branch : mays/mayss_contribution
#  Purpose : real-time webcam demo 
# Status : In Progress
import cv2
import time
import torch
import numpy as np
import mediapipe as mp
from PIL import Image

from scripts.emotion_model import ResNetEmotionModel, EMOTION_DICT
from scripts.gradcamEAI import load_model, compute_gradcam, transform

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mpfacemesh = mp.solutions.face_mesh
facemesh = mpfacemesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1)

# Open a connection to the webcam
cap_webcam = cv2.VideoCapture(0)

# Load the pre-trained emotion recognition model
base_option = Python.BaseOption(model_assest_path='emotion_model.pt')
optipons 
# Grad-CAM & overlay

# frame and showing face box
fps = cap_webcam.get(cv2.CAP_PROP_FPS)
frames_number = 0 

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


# closing the webcam
cap_webcam.release()
cv2.destroyAllWindows()