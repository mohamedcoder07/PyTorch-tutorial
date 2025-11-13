import time
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
from torch import optim

import torchvision.transforms as T


class Model_cnn(nn.Module):
    def __init__(self, num_classes):
        super(Model_cnn, self).__init__()

        # === Convolutional feature extractor ===
        self.conv1 = nn.Conv2d(in_channels = 1 , out_channels = 32 , kernel_size = 4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels = 32 , out_channels = 64 , kernel_size = 4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # === Fully connected classifier ===
        self.fc1 = nn.Linear(1024, 300)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(300, num_classes)

    def forward(self, image):
        # First convolution block
        out = self.conv1(image)
        out = self.pool1(out)     

        # Second convolution block
        out = self.conv2(out)
        out = self.pool2(out)

        # Flatten before passing to fully connected layers
        out = torch.flatten(out, 1)

        # Fully connected layers
        out = self.fc1(out)
        out = self.relu2(out)
        out = self.dropout1(out)
        out = self.fc3(out)

        return out


def train_eval_model(model, train_dataloader, val_dataloader, num_epochs, optimizer, loss_fn, device, verbose=False):
    """
    Train and evaluate a given PyTorch model using the provided dataloaders.

    Parameters:
    - model: neural network to train
    - train_dataloader: DataLoader containing training data
    - val_dataloader: DataLoader containing validation data
    - num_epochs: number of training epochs
    - optimizer: optimizer used for gradient descent
    - loss_fn: loss function
    - device: 'cpu' or 'cuda'
    - verbose: if True, prints progress during training

    Returns:
    - trained model
    - list of training losses per epoch
    - list of validation losses per epoch
    """
    
    # model = deepcopy(model)
    train_losses = []
    val_losses = []

    training_start_time = time.time()

    for epoch in range(num_epochs):
        train_loss = []
        val_loss = []

        # === Training phase ===
        model.train()

        if verbose:
            print(f'----------------------------------\nEpoch [{epoch+1}/{num_epochs}]\n----------------------------------')    

        # Iterate over training batches
        for batch, (X, y) in enumerate(train_dataloader):
            X = X.to(device)
            y = y.to(device)

            # Forward pass
            optimizer.zero_grad()
            train_pred = model(X)
            loss = loss_fn(train_pred, y)
            train_loss.append(loss.item())

            # Backward pass and parameter update
            loss.backward()
            optimizer.step()

            # Periodic training feedback
            if verbose:
                if batch % 200 == 0:
                    print(f'train loss : {loss.item()} [{batch}/{len(train_dataloader)}]')

        train_losses.append(np.mean(train_loss))

        # === Validation phase ===
        model.eval()
        with torch.no_grad():
            for X, y in val_dataloader:
                X, y = X.to(device), y.to(device)
                val_pred = model(X)
                loss = loss_fn(val_pred, y)
                val_loss.append(loss.item())

        val_losses.append(np.mean(val_loss))

        if verbose:
            print(f'\nAvg Train Loss: {np.mean(val_loss):.4f} | Avg Val Loss: {np.mean(train_loss):.4f}')

    # === Summary ===
    duration = time.time() - training_start_time
    print(f'Total training time : {duration // 60:.0f} minutes and {duration % 60:.2f} seconds')

    return model, train_losses, val_losses
