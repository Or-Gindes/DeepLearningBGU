import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from earlyStopping import EarlyStopping
from prepareDataset import PrepareDataset
from torchDataloader import FacesDataLoader
from siameseNetwork import SiameseNetwork

DATASET_FOLDER = "lfw2"
LEARNING_RATE = 5e-4
BATCH_SIZE = 128
EPOCHS = 200
PATIENCE = 20
LAMBDA = 0.95
L2_REG = 1e-5


def main():
    ds = PrepareDataset(directory=os.path.join(os.getcwd(), DATASET_FOLDER))
    train_image_pairs, train_labels, validation_image_pairs, validation_labels = ds.load_dataset(
        file_path=os.path.join(os.getcwd(), "pairsDevTrain.txt")
    )
    train_dataset = FacesDataLoader(
        images=train_image_pairs,
        labels=train_labels,
        transform=transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    validation_dataset = FacesDataLoader(
        images=validation_image_pairs,
        labels=validation_labels,
        transform=transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])
    )
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SiameseNetwork()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REG)
    criterion = torch.nn.BCEWithLogitsLoss()
    lambda_ = lambda epoch: LAMBDA
    scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda_)
    early_stopping = EarlyStopping(patience=PATIENCE)
    model.train_model(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        epoch=EPOCHS,
        optimizer=optimizer,
        loss_criterion=criterion,
        scheduler=scheduler,
        early_stopping=early_stopping
    )

    test_image_pairs, test_labels = ds.load_dataset(file_path=os.path.join(os.getcwd(), "pairsDevTest.txt"), mode="test")
    test_dataset = FacesDataLoader(
        images=test_image_pairs,
        labels=test_labels,
        transform=transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])
    )
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loss, test_auc = model.evaluate_model(test_dataloader, criterion)
    print(f"Test loss: {test_loss:.3f} | Test Auc: {test_auc:.3f}")


if __name__ == "__main__":
    main()
