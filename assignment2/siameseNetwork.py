import torch
import numpy as np
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from earlyStopping import EarlyStopping

np.random.seed(1)
torch.manual_seed(1)
device = ("cuda" if torch.cuda.is_available() else "cpu")
l2_lambda = 0.00001


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Part 1 - Defined separately as to enable running forward pass twice
        self.separate_run = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=10, stride=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(0.25),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7, stride=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(0.25),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            #nn.Dropout(0.25),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  
            #nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(2304, 4096),
            nn.Sigmoid()
            # ,nn.Linear(4096, 4096),
            # nn.Sigmoid(),
            # nn.Linear(4096, 2048),
            # nn.Sigmoid(),
            # nn.Linear(2048, 4096),
            # nn.Sigmoid()
        )
        
        # Part 2 - Joins separate runs and outputs a single prediction
        self.joined_run = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )
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
                torch.nn.init.normal_(l.weight, mean=0, std=0.01)
                torch.nn.init.normal_(l.bias, mean=0.5, std=0.01)
            # Linear
            if isinstance(l, nn.Linear):
                torch.nn.init.normal_(l.weight, mean=0, std=2 * 0.1)
                torch.nn.init.normal_(l.bias, mean=0.5, std=0.01)

    def forward(self, x1, x2) -> torch.Tensor:
        x1 = self.separate_run(x1)
        x2 = self.separate_run(x2)
        l1_distance = torch.abs(x1 - x2)
        pred = self.joined_run(l1_distance)
        return pred

    def loss(self, prediction, label) -> float:
        cross_entropy_loss = nn.BCELoss()
        return cross_entropy_loss(prediction, label) 
        
    def train_model(self, train_dataloader: DataLoader, validation_dataloader: DataLoader, epoch: int, optimizer: torch.optim, scheduler, early_stopping) -> torch.Tensor:
        self.train()
        i = 0
        lossCE = nn.BCELoss()
        while i < epoch: 
            train_loss = 0.
            train_accuracy = 0
            train_predictions = []
            train_labels = []
            for x1, x2, label in train_dataloader:
                x1, x2, label = x1.to(device), x2.to(device), label.to(device)
                assert torch.all(label >= 0) and torch.all(label <= 1)
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
            train_accuracy = roc_auc_score(train_labels, train_predictions) #train_accuracy/len(train_dataloader)
            
            
            validation_loss, validation_accuracy = self.evaluate_model(validation_dataloader)
            print(f'Epoch {i + 1}, Train Loss: {train_loss:.3f}, Train ROC-AUC: {train_accuracy:.3f}, Val Loss: {validation_loss:.3f}, Val ROC-AUC: {validation_accuracy:.3f}')
            early_stopping(validation_loss, self)
            if early_stopping.early_stop:
                print("Early stopping: No improvement in validation loss for 20 epochs.")
                self.load_state_dict(torch.load(f'model_val_loss_{validation_loss}.pt'))
                break
                
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