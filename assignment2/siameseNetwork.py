import torch
import numpy as np
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

np.random.seed(1)
torch.manual_seed(1)
device = ("cuda" if torch.cuda.is_available() else "cpu")
l2_lambda = 0.00001


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Convolutional layers
        self.layer1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=10, stride=1)
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1)
        self.layer3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=1)
        self.layer4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1)
        
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
        with torch.no_grad():
            if isinstance(l, nn.Conv2d):
                torch.nn.init.trunc_normal_(l.weight, mean=0, std=0.01)
                torch.nn.init.trunc_normal_(l.bias, mean=0.5, std=0.01)
            # Linear
            if isinstance(l, nn.Linear):
                torch.nn.init.trunc_normal_(l.weight, mean=0, std=2 * 0.1)
                torch.nn.init.trunc_normal_(l.bias, mean=0.5, std=0.01)


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

    def loss(self, prediction, label) -> float:
        cross_entropy_loss = nn.BCELoss()
        return cross_entropy_loss(prediction, label) 
        
    def train_model(self, train_dataloader: DataLoader, validation_dataloader: DataLoader, epoch: int, optimizer: torch.optim) -> torch.Tensor:
        self.train()
        i = 0
        lossCE = nn.BCELoss()
        while i < epoch:  # and ():
            train_loss = 0.
            train_accuracy = 0
            train_predictions = []
            train_labels = []
            for x1, x2, label in train_dataloader:
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                prediction = self.forward(x1, x2).squeeze()
                loss = lossCE(prediction, label.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_accuracy += torch.sum(torch.all(prediction >= 0.5) == label)
                train_predictions.append(prediction.detach().cpu().numpy().squeeze())
                train_labels.append(label.cpu().numpy().squeeze())

            train_predictions = np.concatenate(train_predictions) # flatten train_predictions
            train_labels = np.concatenate(train_labels) # flatten labels
            train_loss /= len(train_dataloader)
            train_accuracy =  roc_auc_score(train_labels, train_predictions)#train_accuracy/len(train_dataloader)
            
            
            validation_loss, validation_accuracy = self.evaluate_model(validation_dataloader)
            print(f'Epoch {i + 1}, Train Loss: {train_loss:.3f}, Train ROC-AUC: {train_accuracy:.3f}, Val Loss: {validation_loss:.3f}, Val ROC-AUC: {validation_accuracy:.3f}')
            i += 1

    def evaluate_model(self, validation_dataloader: DataLoader):
        self.eval()
        loss = 0.
        accuracy = 0
        predictions = []
        labels = []
        with torch.no_grad():
            for x1, x2, label in validation_dataloader:
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                prediction = self.forward(x1, x2).squeeze()
                loss += self.loss(prediction=prediction, label=label.float())
                predictions.append(prediction.cpu().detach().numpy().squeeze())
                labels.append(label.cpu().numpy().squeeze())
        
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        loss /= len(validation_dataloader)
        accuracy = roc_auc_score(labels, predictions)
        
        return loss, accuracy