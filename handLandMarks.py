import os
import mediapipe
import cv2
import numpy as np
import pandas as pd
from PIL import Image

handTracker = mediapipe.solutions.hands # Initialize the MediaPipe Hands class for hand tracking
handDetector = handTracker.Hands(static_image_mode=True, max_num_hands=2,  min_detection_confidence= 0.2 ) # hold the landmarks points# Configure the MediaPipe Hands instance for detecting hands

DATA_FOLDER = '../datasets/asl_alphabet_test' # Alphabet test
os.environ["PYTHONIOENCODING"] = "utf-8"

coordinates = []  # List to store data for all characters
index = 0

for file in os.listdir(DATA_FOLDER):
    for imgPath in os.listdir(os.path.join(DATA_FOLDER, str(file))):
        fullImgPath = os.path.join(DATA_FOLDER, file, imgPath).replace('\\', '/')
        image = Image.open(fullImgPath)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        #imgResize = cv2.resize(img, (256, 256))        # Resize the image to 256x256 pixels

        if img is None:
            print(f"Failed to load image from {fullImgPath}")
            continue  # Skip this iteration if the image failed to load


        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        imgMediapipe = handDetector.process(imgRGB)

        x_Coordinates = []
        y_Coordinates = []
        z_Coordinates = []

        if imgMediapipe.multi_hand_landmarks:
            for handLandmarks in imgMediapipe.multi_hand_landmarks:
                data = {}
                data['CHARACTER'] = file
                data['GROUPVALUE'] = index

                for i in range(len(handLandmarks.landmark)):
                    lm = handLandmarks.landmark[i]
                    x_Coordinates.append(lm.x)
                    y_Coordinates.append(lm.y)
                    z_Coordinates.append(lm.z)

                for i, landmark in enumerate(handTracker.HandLandmark): # Apply Min-Max normalization
                    lm = handLandmarks.landmark[i]
                    data[f'{landmark.name}_x'] = lm.x - min(x_Coordinates)
                    data[f'{landmark.name}_y'] = lm.y - min(y_Coordinates)
                    data[f'{landmark.name}_z'] = lm.z - min(z_Coordinates)
                coordinates.append(data)
    index+=1

df = pd.DataFrame(coordinates) # Convert to DataFrame

excel_path = "asl_alphabet_testing_data.xlsx"
df.to_excel(excel_path, index=False)