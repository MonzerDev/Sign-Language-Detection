import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from CNN.CNNModel import CNNModel
from torch.optim import Adam
from torch.nn import CrossEntropyLoss


# a function to move data to CUDA if available
def to_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def calculateAccuracy(y_true, y_pred):
    # If y_true is one-hot encoded, convert it to class indices
    if y_true.dim() > 1 and y_true.size(1) > 1:
        y_true = torch.argmax(y_true, dim=1)

    y_pred = y_pred.to(y_true.device)

    predicted_classes = torch.argmax(y_pred, dim=1)
    correct_predictions = (predicted_classes == y_true).float()
    accuracy = correct_predictions.sum() / len(correct_predictions)
    return accuracy


# Function to plot accuracy graph
def plotAccuracyGraph(trainAccuracies, valAccuracies, epoch):
    plt.plot(range(1, epoch + 1), trainAccuracies, 'bo-', label='Training Accuracy')
    plt.plot(range(1, epoch + 1), valAccuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# Function to plot loss graph
def plotLossGraph(trainLosses, valLosses):
    plt.plot(trainLosses, label='Training loss')
    plt.plot(valLosses, label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Load data
data = pd.read_excel("../featureExtraction/alphabet_data.xlsx", header=0)

data.pop("CHARACTER")
groupValue, coordinates = data.pop("GROUPVALUE"), data.copy()
coordinates = np.reshape(coordinates.values, (coordinates.shape[0], 63, 1))
coordinates = torch.from_numpy(coordinates).float()
groupValue = torch.from_numpy(groupValue.to_numpy()).long()

k_folds = 4
epoch = 70
foldTrainLosses = []
foldValLosses = []
foldTrainAccuracies = []
foldValAccuracies = []
foldTrainPrecision = []
foldValPrecision = []
foldTrainRecall = []
foldValRecall = []
foldTrainF1 = []
foldValF1 = []

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

for fold, (trainIndex, valIndex) in enumerate(kf.split(coordinates)):
    print(f"Training on fold {fold + 1}/{k_folds}")

    # Splitting data for current fold
    training = coordinates[trainIndex]
    groupValueTraining = groupValue[trainIndex]

    validation = coordinates[valIndex]
    groupValueValidation = groupValue[valIndex]

    # Model and optimization setup
    model = CNNModel()
    model = to_cuda(model)
    # summary(model, input_size=(63, 1))
    optimizer = Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    criterion = CrossEntropyLoss()

    trainLosses = []
    valLosses = []
    trainAccuracies = []
    valAccuracies = []
    trainPrecisions = []
    trainRecalls = []
    trainF1s = []
    valPrecisions = []
    valRecalls = []
    valF1s = []

    for epochi in range(1, epoch + 1):
        model.train()
        optimizer.zero_grad()

        # Move  to CUDA
        training = to_cuda(training)
        validation = to_cuda(validation)
        groupValueTraining = to_cuda(groupValueTraining)
        groupValueValidation = to_cuda(groupValueValidation)

        outputTrain = model(training)
        outputVal = model(validation)

        lossTrain = criterion(outputTrain, groupValueTraining)
        trainLosses.append(lossTrain.item())

        lossVal = criterion(outputVal, groupValueValidation)
        valLosses.append(lossVal.item())

        lossTrain.backward()
        optimizer.step()

        # Switch to evaluation mode for accuracy calculation
        with torch.no_grad():
            training = to_cuda(training)
            validation = to_cuda(validation)

            # Calculate training accuracy
            trainOutput = model(training).cpu()
            trainAccuracy = calculateAccuracy(groupValueTraining, trainOutput)
            trainAccuracies.append(trainAccuracy.item())

        model.eval()
        with torch.no_grad():
            # Calculate validation accuracy
            outputValid = model(validation).cpu()
            validAccuracy = calculateAccuracy(groupValueValidation, outputValid)
            valAccuracies.append(validAccuracy.item())

        # Calculate metrics for training data
        trainPrec = precision_score(groupValueTraining.cpu(), torch.argmax(trainOutput, dim=1), average='weighted',
                                    zero_division=0)
        trainRec = recall_score(groupValueTraining.cpu(), torch.argmax(trainOutput, dim=1), average='weighted',
                                zero_division=0)
        trainF1 = f1_score(groupValueTraining.cpu(), torch.argmax(trainOutput, dim=1), average='weighted',
                           zero_division=0)

        # Calculate metrics for validation data
        valPrec = precision_score(groupValueValidation.cpu(), torch.argmax(outputValid, dim=1), average='weighted',
                                  zero_division=0)
        valRec = recall_score(groupValueValidation.cpu(), torch.argmax(outputValid, dim=1), average='weighted',
                              zero_division=0)
        valF1 = f1_score(groupValueValidation.cpu(), torch.argmax(outputValid, dim=1), average='weighted',
                         zero_division=0)

        # Store metrics
        trainPrecisions.append(trainPrec)
        trainRecalls.append(trainRec)
        trainF1s.append(trainF1)
        valPrecisions.append(valPrec)
        valRecalls.append(valRec)
        valF1s.append(valF1)

        if epochi % 10 == 0:
            print(
                f'Fold: {fold + 1}, Epoch : {epochi}, Training Loss: {lossTrain.item()}, Validation Loss: {lossVal.item()}')

    foldTrainLosses.append(trainLosses)
    foldValLosses.append(valLosses)
    foldTrainAccuracies.append(trainAccuracies)
    foldValAccuracies.append(valAccuracies)
    # Store fold-wise metrics
    foldTrainPrecision.append(trainPrecisions)
    foldTrainRecall.append(trainRecalls)
    foldTrainF1.append(trainF1s)
    foldValPrecision.append(valPrecisions)
    foldValRecall.append(valRecalls)
    foldValF1.append(valF1s)
    # print(f'Training Accuracy: {trainAccuracy * 100:.2f}%   Validation Accuracy for fold {fold + 1}: {validAccuracy * 100:.2f}%\n')

# Average loss and accuracy
avgTrainLoss = np.mean(foldTrainLosses,
                       axis=0)  # Using axis=0 with these arrays calculates the mean across the same epoch across different folds
avgValLoss = np.mean(foldValLosses, axis=0)
avgTrainAccuracy = np.mean(foldTrainAccuracies, axis=0)
avgValAccuracy = np.mean(foldValAccuracies, axis=0)
# Calculate and print average metrics
avgTrainPrec = np.mean(foldTrainPrecision, axis=0)
avgTrainRec = np.mean(foldTrainRecall, axis=0)
avgTrainF1 = np.mean(foldTrainF1, axis=0)
avgValPrec = np.mean(foldValPrecision, axis=0)
avgValRec = np.mean(foldValRecall, axis=0)
avgValF1 = np.mean(foldValF1, axis=0)
# Plotting loss and accuracy graphs
plotLossGraph(avgTrainLoss, avgValLoss)

plotAccuracyGraph(avgTrainAccuracy, avgValAccuracy, epoch)

# Final average accuracy
finalAvgTrainAccuracy = avgTrainAccuracy[
    -1]  # refers to the last element of that list, evaluate the final performance after completing all training epochs
finalAvgValAccuracy = avgValAccuracy[-1]
print(f"Final Average Training Accuracy: {finalAvgTrainAccuracy * 100}")
print(f"Final Average Validation Accuracy: {finalAvgValAccuracy * 100}")

print(f"Final Average Training Precision: {avgTrainPrec[-1] * 100}")
print(f"Final Average Validation Precision: {avgValPrec[-1] * 100}")

print(f"Final Average Training Recall: {avgTrainRec[-1] * 100}")
print(f"Final Average Validation Recall: {avgValRec[-1] * 100}")

print(f"Final Average Training F1-Score: {avgTrainF1[-1] * 100}")
print(f"Final Average Validation F1-Score: {avgValF1[-1] * 100}")

# Save the model
modelPath = "CNN_model_alphabet_SIBI.pth"
# modelPath = "CNN_model_number_SIBI.pth"
# modelPath = "CNN_model_asl_alphabet_SIBI.pth"

torch.save(model.state_dict(), modelPath)
