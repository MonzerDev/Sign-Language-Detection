import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from CNN.CNNModel import CNNModel

# Plot the confusion matrix
def plotConfusionMatrix(confusionMatrix,classNames):
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusionMatrix, annot=True, fmt='g', xticklabels= classNames, yticklabels= classNames)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

# Load the model
model = CNNModel()
model.load_state_dict(torch.load("CNN_model_alphabet_SIBI.pth"))

model.eval()

# Load data
data = pd.read_excel("../featureExtraction/alphabet_testing_data.xlsx", header=0)
#data = pd.read_excel("../featureExtraction/numbers_testing_data.xlsx", header=0)
#data = pd.read_excel("../featureExtraction/asl_alphabet_testing_data.xlsx", header=0)
data.pop("CHARACTER")
groupValue, coordinates = data.pop("GROUPVALUE"), data.copy()
coordinates = np.reshape(coordinates.values, (coordinates.shape[0], 63, 1))
coordinates = torch.from_numpy(coordinates).float()
coordinates = [coordinates]
groupValue = groupValue.to_numpy()

predictions = []

with torch.no_grad():
    for inputs in coordinates:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())

accuracy = accuracy_score(predictions,groupValue)
precision = precision_score(groupValue, predictions, average='weighted', zero_division=0)
recall = recall_score(groupValue, predictions, average='weighted', zero_division=0)
f1 = f1_score(groupValue, predictions, average='weighted', zero_division=0)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision* 100:.2f}")
print(f"Recall: {recall* 100:.2f}")
print(f"F1-Score: {f1* 100:.2f}")

classNames = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
#classNames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


# Compute the confusion matrix
confusionMatrix = confusion_matrix(groupValue, predictions)

plotConfusionMatrix(confusionMatrix,classNames)