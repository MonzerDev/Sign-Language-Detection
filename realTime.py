import cv2
import mediapipe
import torch
import numpy as np
import pandas as pd
from CNNModel import CNNModel

# Load the model
model = CNNModel()
model.load_state_dict(torch.load("CNN_model_alphabet_SIBI.pth"))

cap = cv2.VideoCapture(0)

handTracker = mediapipe.solutions.hands
drawing = mediapipe.solutions.drawing_utils
drawingStyles = mediapipe.solutions.drawing_styles

handDetector = handTracker.Hands(static_image_mode=True, min_detection_confidence= 0.2 ) # hold the landmarks points# Configure the MediaPipe Hands instance for detecting hands

classes = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
    'K': 10,
    'L': 11,
    'M': 12,
    'N': 13,
    'O': 14,
    'P': 15,
    'Q': 16,
    'R': 17,
    'S': 18,
    'T': 19,
    'U': 20,
    'V': 21,
    'W': 22,
    'X': 23,
    'Y': 24,
    'Z': 25
}

model.eval()

while True:

    ret, frame = cap.read()
    #frame = cv2.flip(frame, 1)

    height , width, _ = frame.shape

    frameRGB= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    imgMediapipe = handDetector.process(frameRGB)

    coordinates = []
    x_Coordinates = []
    y_Coordinates = []
    z_Coordinates = []

    if imgMediapipe.multi_hand_landmarks:
        for handLandmarks in imgMediapipe.multi_hand_landmarks:
            drawing.draw_landmarks(
                frame,  # image to draw
                handLandmarks,  # model output
                handTracker.HAND_CONNECTIONS,  # hand connections
                drawingStyles.get_default_hand_landmarks_style(),
                drawingStyles.get_default_hand_connections_style())

            data = {}

            for i in range(len(handLandmarks.landmark)):
                lm = handLandmarks.landmark[i]
                x_Coordinates.append(lm.x)
                y_Coordinates.append(lm.y)
                z_Coordinates.append(lm.z)

            for i, landmark in enumerate(handTracker.HandLandmark):  # Apply Min-Max normalization
                lm = handLandmarks.landmark[i]
                data[f'{landmark.name}_x'] = lm.x - min(x_Coordinates)
                data[f'{landmark.name}_y'] = lm.y - min(y_Coordinates)
                data[f'{landmark.name}_z'] = lm.z - min(z_Coordinates)
            coordinates.append(data)

        x1 = int(min(x_Coordinates) * width) - 10
        y1 = int(min(y_Coordinates) * height) - 10

        x2 = int(max(x_Coordinates) * width) - 10
        y2 = int(max(y_Coordinates) * height) - 10

        predictions = []
        coordinates = pd.DataFrame(coordinates)
        coordinates = np.reshape(coordinates.values, (coordinates.shape[0], 63, 1))
        coordinates = torch.from_numpy(coordinates).float()

        with torch.no_grad():
            outputs = model(coordinates)
            _, predicted = torch.max(outputs.data, 1)
            predictions = predicted.cpu().numpy()

        predicted_character = [key for key, value in classes.items() if value == predictions[0]][0]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit the loop if 'q' is pressed

cap.release()
cv2.destroyAllWindows()