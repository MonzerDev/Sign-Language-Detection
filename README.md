Real-Time Sign Language Detection

This project implements a real-time sign language detection system using a Convolutional Neural Network (CNN) and MediaPipe for hand landmark detection. The system captures live video input, processes hand gestures, and classifies them into corresponding sign language alphabets.

Project Structure

- `CNNModel.py`: Defines the Convolutional Neural Network (CNN) architecture used for classifying hand gestures.
- `handLandMarks.py`: Handles the detection of hand landmarks using MediaPipe and processes them for use by the CNN model.
- `mediapipeHandDetection.py`: Integrates MediaPipe to perform real-time hand detection through the webcam.
- `realTime.py`: The main script that ties everything together, using the CNN model and MediaPipe for real-time sign language detection.
- `training.py`: Script used for training the CNN model on a dataset of hand gestures.
- `testCNN.py`: Script for testing the performance of the trained CNN model on a test dataset.
- `CNN_model_alphabet_SIBI.pth`: Pre-trained CNN model weights used for classification.

How to Run the Project

1. Install Dependencies

Make sure you have Python installed on your system. You can install the required Python packages using pip:

pip install -r requirements.txt

If you don't have a `requirements.txt` file, you can manually install the necessary packages:

pip install opencv-python mediapipe torch numpy pandas

2. Running Real-Time Detection

To start the real-time sign language detection, run the following command:

python realTime.py

This will activate your webcam and start detecting and classifying hand gestures in real-time.

3. Training the Model (Optional)

If you want to train the CNN model from scratch, you can run:

python training.py

This script will use a dataset of hand gestures to train the model.

4. Testing the Model (Optional)

To test the performance of the trained CNN model on a test dataset, you can run:

python testCNN.py

How It Works

1. Hand Landmark Detection: 
   - The system uses MediaPipe to detect and track hand landmarks in real-time from the webcam feed.

2. Feature Extraction:
   - The detected hand landmarks are processed and used as input features for the CNN model.

3. Gesture Classification:
   - The CNN model classifies the input features into one of the predefined sign language alphabets (A-Z).

4. Real-Time Feedback:
   - The classified gesture is displayed in real-time, providing immediate feedback to the user.

Requirements

- Python 3.x
- OpenCV
- MediaPipe
- PyTorch
- Pandas

Future Improvements

- Extend the Model: Support for additional sign language gestures or other hand-based communication systems.
- Optimize Performance: Improve the real-time performance for better accuracy and responsiveness.
- Deploy the System: Deploy the detection system as a web application or mobile app.

Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand tracking and landmark detection.
- [PyTorch](https://pytorch.org/) for deep learning model implementation.

Contact

For any questions or suggestions, please feel free to contact me at [monzerkoukou@gmail.com].
