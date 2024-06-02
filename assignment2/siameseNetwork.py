import torch
import numpy as np
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

np.random.seed(0)
torch.manual_seed(0)
device = ("cuda" if torch.cuda.is_available() else "cpu")
l2_lambda = 0.01


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Convolutional layers
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=10, stride=1)
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1)
        self.layer3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=1)
        self.layer4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1)
        # self.layer1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=9, stride=1, padding_mode='valid')
        # self.layer2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=8, stride=1, padding_mode='valid'),
        # self.layer3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=1, padding_mode='valid'),
        # self.layer4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding_mode='valid'),
        # Dense/Linear layers
        self.layer5 = nn.Linear(6 * 6 * 256, 4096)  # nn.Linear(24 * 24 * 256, 4096),
        self.layer6 = nn.Linear(4096, 1)

        # Pooling and Activation layers
        self.flatten = nn.Flatten()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Initialize layers' weights and biases
        self.apply(self.initialize_weights)


    def initialize_weights(self, l):
        """
        Initializes the model's layers' weights
        """
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
        """
        runs pre-jointed processing for a single sub-network at a time
        :param x: input image
        :return: processed outputs
        """
        x = self.relu(self.layer1(x))
        x = self.maxPool(x)
        x = self.relu(self.layer2(x))
        x = self.maxPool(x)
        x = self.relu(self.layer3(x))
        x = self.maxPool(x)
        x = self.relu(self.layer4(x))
        x = self.flatten(x)
        x = self.sigmoid(self.layer5(x))
        return x

    def forward(self, x1, x2) -> torch.Tensor:
        x1 = self.separated_forward(x1)
        x2 = self.separated_forward(x2)
        l1_distance = torch.abs(x1 - x2)
        return self.sigmoid(self.layer6(l1_distance))

    def loss(self, prediction, label):
        cross_entropy_loss = label * torch.log(prediction) + (1 - label) * torch.log(1 - prediction)
        l2 = sum(torch.sum(torch.square(weights)) for weights in self.parameters())
        return torch.mean(-1 * cross_entropy_loss) + l2_lambda * l2

    def train_model(self, train_dataloader: DataLoader, validation_dataloader: DataLoader, epoch: int, learning_rate: float) -> torch.Tensor:
        self.train()
        i = 0
        train_loss = 0.
        validation_loss = 0.
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        while i < epoch:  # and ():
            for x1, x2, label in train_dataloader:
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                optimizer.zero_grad()
                prediction = self.forward(x1, x2)
                loss = self.loss(prediction=prediction, label=label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_dataloader)
            with torch.no_grad():
                for x1, x2, label in validation_dataloader:
                    x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                    validation_prediction = self.forward(x1, x2)
                    loss = self.loss(validation_prediction, label)
                    validation_loss += loss.item()

                validation_loss /= len(validation_dataloader)
                print(f'Epoch {i + 1}, Train Loss: {train_loss:.3f}, Val Loss: {validation_loss:.3f}')
            i += 1
