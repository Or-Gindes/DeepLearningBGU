import torch
import numpy as np
from torch import nn, Tensor
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

np.random.seed(1)
torch.manual_seed(1)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=10),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.liner = nn.Sequential(nn.Linear(in_features=9216, out_features=4096), nn.Sigmoid())
        self.out = nn.Linear(in_features=4096, out_features=1)

    def embedding(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2) -> torch.Tensor:
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        l1_distance = torch.abs(x1 - x2)
        pred = torch.sigmoid(self.out(l1_distance))
        return pred

    def train_model(
            self,
            train_dataloader: DataLoader,
            validation_dataloader: DataLoader,
            epoch: int,
            optimizer: torch.optim,
            loss_criterion,
            scheduler,
            early_stopping
    ) -> dict[str, list[Tensor] | list[float]]:
        self.to(self.device)
        self.train()
        loss_criterion.to(self.device)
        train_losses = []
        validation_losses = []
        train_aucs = []
        validation_aucs = []

        for epoch in range(epoch):
            train_loss = torch.tensor(0.0, device=self.device)
            train_auc = torch.tensor(0.0, device=self.device)
            train_predictions = []
            train_labels = []
            for x1, x2, label in train_dataloader:
                x1, x2, label = x1.to(self.device), x2.to(self.device), label.to(self.device)
                assert torch.all(label >= 0) and torch.all(label <= 1)
                prediction = self.forward(x1, x2)   #.squeeze()
                loss = loss_criterion(prediction, label.unsqueeze(1).float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_auc += torch.sum(torch.all(prediction >= 0.5) == label)
                train_predictions.append(prediction.detach().cpu().numpy().squeeze())
                train_labels.append(label.cpu().numpy().squeeze())

            train_predictions = np.concatenate(train_predictions)  # flatten train_predictions
            train_labels = np.concatenate(train_labels)  # flatten labels
            train_loss /= len(train_dataloader)
            train_auc = roc_auc_score(train_labels, train_predictions)

            validation_loss, validation_auc = self.evaluate_model(validation_dataloader, loss_criterion)
            print(
                f'Epoch {epoch}, Train Loss: {train_loss:.3f}, Train ROC-AUC: {train_auc:.3f}, '
                f'Val Loss: {validation_loss:.3f}, Val ROC-AUC: {validation_auc:.3f}'
            )

            # Store training history
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            train_aucs.append(train_auc)
            validation_aucs.append(validation_auc)

            # Early Stopping
            early_stopping(epoch, validation_loss, self)
            if early_stopping.early_stop:
                print("Early stopping: No improvement in validation loss for 20 epochs.")
                self.load_state_dict(torch.load(f'model_checkpnt.pt'))
                break

        best_model_ind = early_stopping.epoch
        print(
            f'Finish Training, stats of best model:\n'
            f'Epochs {best_model_ind}, Train Loss: {train_losses[best_model_ind]:.3f}, '
            f'Train ROC-AUC: {train_aucs[best_model_ind]:.3f}, Val Loss: {validation_losses[best_model_ind]:.3f}, '
            f'Val ROC-AUC: {validation_aucs[best_model_ind]:.3f}')

        return {
            "train loss": train_losses[:best_model_ind],
            "validation loss": validation_losses[:best_model_ind],
            "train ROC-AUC": train_aucs[:best_model_ind],
            "validation  ROC-AUC": validation_aucs[:best_model_ind]
        }

    def evaluate_model(self, validation_dataloader: DataLoader, loss_criterion):
        self.eval()
        loss = 0.
        predictions = []
        labels = []
        with torch.no_grad():
            for x1, x2, label in validation_dataloader:
                x1, x2, label = x1.to(self.device), x2.to(self.device), label.to(self.device)
                prediction = self.forward(x1, x2)
                loss += loss_criterion(prediction, label.unsqueeze(1).float())
                predictions.append(prediction.cpu().detach().numpy().squeeze())
                labels.append(label.cpu().numpy().squeeze())

        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        loss /= len(validation_dataloader)
        auc = roc_auc_score(labels, predictions)

        return loss, auc
