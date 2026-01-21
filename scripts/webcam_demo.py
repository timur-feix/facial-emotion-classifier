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

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Load the pre-trained emotion recognition model

# Grad-CAM & overlay

# frame and showing face box
# display predicted emotion on the video feed

# closing the webcam
cap.release()
cv2.destroyAllWindows()