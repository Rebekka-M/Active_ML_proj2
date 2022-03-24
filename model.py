import torch
import numpy as np
from torch import nn
# from tqdm import tqdm, trange
from lauges_tqdm import tqdm
from vars import *

class CNN_class(nn.Module):
    """
    A simple CNN model to classify images of default shape 28x28 and default 62 classes
    """
    def __init__(self, width, depth, input_features=28, n_classes=62):
        """
        Defines the structure of the network.
        Each layer i in [0, depth] has width*(2**i) output channels from 3x3 kernels
        After each layer there is a dropout layer.

        Parameters
        ----------
        width : int, the width parameter of the network (default from 1 to 4)
        depth : int, amount of convolutional layers (default from 1 to 4)
        input_features : int, size of the quadratic image
        n_classes : int, number of classes
        """
        super().__init__()

        self.input_features = input_features
        self.width = int(width)
        self.depth = int(depth)

        # First convolutional layer
        layers = []
        layers.append(nn.Conv2d(1, self.width * 2, kernel_size=3, padding=1))
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.ReLU(inplace=True))

        # The other convolutional layers
        for i in range(2, self.depth+1):
            layers.append(nn.Conv2d(self.width*2**(i-1), self.width*2**i, kernel_size=3, padding=1))
            layers.append(nn.MaxPool2d(2))
            layers.append(nn.ReLU(inplace=True))

        # Calculate the shape of the output of the CNN, used for the linear layer
        temp = self.input_features
        for i in range(self.depth):
            temp = temp // 2

        self.CNN = nn.Sequential(*layers)

        # The linear layer that maps the flattened last CNN feature to the classes.
        self.linear = nn.Linear(self.width*2**self.depth*temp**2, n_classes)

    def forward(self, x):
        """
        Applies the network to the input image batch x

        Parameters
        ----------
        x : (batch_size, 1, input_features, input_features) a batch of images as tensors

        Returns
        -------
        List of length (batch_size, amount_classes), the larger the value the more likely it is that
        class from the models perspective.
        To get pseudo probabilities add a soft max to this.
        """
        x = self.CNN(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x


def train(model, dataloader, lr, weight_decay, n_epochs=10):
    """
    Trains the network for n_epochs with the data set in dataloader.

    Parameters
    ----------
    model : (CNN_class) instance of CNN_class model.
    dataloader : Dataloader that contains tensors of the shape (1, 28, 28) and corresponding labels
    lr : learning rate
    weight_decay : weight decay =)
    n_epochs : number of epochs

    Returns
    -------

    """
    # Optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    # Training loop
    for epoch in tqdm(range(n_epochs)):
        for im, label in tqdm(dataloader):
            # Image needs to be transposed
            im = torch.permute(im, (0, 1, 3, 2))
            optimizer.zero_grad()
            im = im.to(DEVICE)
            label = label.to(DEVICE)

            preds = model(im)
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()


def test(model, dataloader):
    """
    Returns the test accuracy of the model with the dataset in dataloader

    Parameters
    ----------
    model : the model that needs to be tested
    dataloader : Dataloader that contains tensors of the shape (1, 28, 28) and corresponding labels

    Returns
    -------
    Accuracy
    """
    corrects = []
    model.eval()
    with torch.no_grad():
        for im, label in dataloader:
            im = torch.permute(im, (0, 1, 3, 2))
            im = im.to(DEVICE)
            label = label.to(DEVICE)
            corrects.append(model(im).argmax(dim=1) == label)

        acc = torch.cat(corrects).detach().cpu().numpy().mean()
    return acc
