import os
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

np.random.seed(0)
torch.manual_seed(0)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
l2_lambda = 0.01


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Convolutional layers
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=10, stride=1, padding_mode='valid')
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding_mode='valid'),
        self.layer3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding_mode='valid'),
        self.layer4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding_mode='valid'),
        # self.layer1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding_mode='valid')
        # self.layer2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=8, stride=1, padding_mode='valid'),
        # self.layer3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding_mode='valid'),
        # self.layer4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding_mode='valid'),
        # Dense/Linear layers
        self.layer5 = nn.Linear(6 * 6 * 256, 4096),  # nn.Linear(24 * 24 * 256, 4096),
        self.layer6 = nn.Linear(4096, 1),

        # Pooling and Activation layers
        self.flatten = nn.Flatten()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2),
        self.relu = nn.ReLU(),
        self.sigmoid = nn.Sigmoid()

        # Initialize layers' weights and biases
        self.apply(self.initialize_weights)


    def initialize_weights(l):
        # Initialize layers as appears in the article
        # Conv2D
        if isinstance(l, nn.Conv2d):
            torch.nn.init.normal_(l.weight, mean=0, std=0.01)
            torch.nn.init.normal_(l.bias, mean=0.5, std=0.01)
        # Linear
        if isinstance(l, nn.Linear):
            torch.nn.init.normal_(l.weight, mean=0, std=2 * 0.1)
            torch.nn.init.normal_(l.bias, mean=0.5, std=0.01)

    def separated_forward(self, x: np.ndarray) -> torch.Tensor:
        x = self.relu(self.layer1(x))
        x = self.maxPool(x)
        x = self.relu(self.layer2(x))
        x = self.maxPool(x)
        x = self.relu(self.layer3(x))
        x = self.maxPool(x)
        x = self.flatten(x)
        x = self.relu(self.layer4(x))
        x = self.sigmoid(self.layer5(x))
        return x

    def forward(self, x1, x2) -> torch.Tensor:
        x1 = self.separated_forward(x1)
        x2 = self.separated_forward(x2)
        l1_distance = torch.abs(x1 - x2)
        return self.sigmoid(self.layer6(l1_distance))


    def train(self, train_dataloader) -> torch.Tensor:
        while (() and ()):
            for x1, x2, label in train_dataloader:
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)

                optimizer.zero_grad()

                prediction = self.forward(x1, x2)
                loss = label * torch.log(prediction) + (1 - label) * (1 - torch.log(prediction))
                # + l2_lambda * torch.square(W) #What is this l2 regularization - what is W?



