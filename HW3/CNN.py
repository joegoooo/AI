import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Tuple
import pandas as pd

class CNN(nn.Module):
    def __init__(self, num_classes=5):
        # (TODO) Design your CNN, it can only be less than 3 convolution layers
        super(CNN, self).__init__()
        self.input_shape = 3 # the number of color channel
        self.hidden_unit = 20
        self.output_shape = num_classes
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_shape,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
        )

        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=512*784*4,
                out_features=self.output_shape
            ),
            nn.Softmax(dim=1),
        )


    def forward(self, x):
        # (TODO) Forward the model
        x = self.conv_block(x)
        x = self.classification(x)
        return x

def train(model: CNN, train_loader: DataLoader, criterion, optimizer, device)->float:
    # (TODO) Train the model and return the average loss of the data, we suggest use tqdm to know the progress
    avg_loss = 0
    train_loss = 0
    # compute on device we selected, which GPU
    # set module to train mode
    model.train()
    model.to(device)
    for (X, y) in tqdm((train_loader)):
        X, y = X.to(device), y.to(device)
        # forward pass
        y_pred = model(X)
        # calculus loss
        loss = criterion(y_pred, y)
        train_loss += loss.item()
        # optimizer zero grad
        optimizer.zero_grad()
        # loss backward
        loss.backward()
        # optimizer step
        optimizer.step()
    avg_loss = train_loss / len(train_loader)
    print(f"Train Loss: {avg_loss:.4f}")
    return avg_loss


def validate(model: CNN, val_loader: DataLoader, criterion, device)->Tuple[float, float]:
    # (TODO) Validate the model and return the average loss and accuracy of the data, we suggest use tqdm to know the progress
    model.eval()
    train_loss = 0
    accuracy = 0
    for (X, y) in tqdm((val_loader)):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        train_loss += loss.item()
        for i in range(y.shape[0]):
            if torch.argmax(y_pred[i]) == y[i]:
                accuracy += 1
        
    avg_loss = train_loss / len(val_loader)
    accuracy /= len(val_loader)*32
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss, accuracy

def test(model: CNN, test_loader: DataLoader, criterion, device):
    # (TODO) Test the model on testing dataset and write the result to 'CNN.csv'
    model.eval()
    result = []
    for (X, y) in tqdm((test_loader)):
        X = X.to(device)
        y_pred = model(X)

        for i in range(len(y)):
            result.append({'id': y[i], 'prediction': int(torch.argmax(y_pred[i]))})

    df = pd.DataFrame(result)
    df.to_csv('CNN.csv', index=False)
    print(f"Predictions saved to 'CNN.csv'")
    return


