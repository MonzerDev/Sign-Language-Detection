import cv2
import mediapipe
import numpy as np
import pandas as pd
from joblib import load

cap = cv2.VideoCapture(0)

handTracker = mediapipe.solutions.hands
drawing = mediapipe.solutions.drawing_utils
drawingStyles = mediapipe.solutions.drawing_styles

handDetector = handTracker.Hands(static_image_mode=True, min_detection_confidence= 0.2 ) # hold the landmarks points# Configure the MediaPipe Hands instance for detecting hands


while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    height , width, _ = frame.shape

    frameRGB= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    imgMediapipe = handDetector.process(frameRGB)

    if imgMediapipe.multi_hand_landmarks:
        for handLandmarks in imgMediapipe.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame,  # image to draw
                handLandmarks,  # model output
                handTracker.HAND_CONNECTIONS,  # hand connections
                drawingStyles.get_default_hand_landmarks_style(),
                drawingStyles.get_default_hand_connections_style())


    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop if 'q' is pressed

cap.release()
cv2.destroyAllWindows()