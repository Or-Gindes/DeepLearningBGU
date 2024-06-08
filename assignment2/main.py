import os
import torch
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from earlyStopping import EarlyStopping
from prepareDataset import PrepareDataset
from torchDataloader import FacesDataLoader
from siameseNetwork import SiameseNetwork
from PIL import Image
import matplotlib.pyplot as plt


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
    criterion = torch.nn.BCELoss()
    lambda_ = lambda epoch: LAMBDA
    scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda_)
    early_stopping = EarlyStopping(patience=PATIENCE)
    history = model.train_model(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        epoch=EPOCHS,
        optimizer=optimizer,
        loss_criterion=criterion,
        scheduler=scheduler,
        early_stopping=early_stopping
    )

    # Plot train and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(history["train loss"], label="Train Loss")
    plt.plot(history["validation loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot train and validation ROC-AUC scores
    plt.figure(figsize=(10, 5))
    plt.plot(history["train ROC-AUC"], label="Train ROC-AUC")
    plt.plot(history["validation ROC-AUC"], label="Validation ROC-AUC")
    plt.xlabel("Epochs")
    plt.ylabel("ROC-AUC Score")
    plt.title("Train and Validation ROC-AUC")
    plt.legend()
    plt.grid(True)
    plt.show()

    test_image_pairs, test_labels = ds.load_dataset(file_path=os.path.join(os.getcwd(), "pairsDevTest.txt"), mode="test")
    test_dataset = FacesDataLoader(
        images=test_image_pairs,
        labels=test_labels,
        transform=transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])
    )
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loss, test_auc, test_accuracy, test_pred = model.evaluate_model(test_dataloader, criterion)
    print(f"Test loss: {test_loss:.3f} | Test Auc: {test_auc:.3f} | Test Accuracy: {test_accuracy:.3f}")

    correct_preds = random.sample(np.where((test_pred >= 0.5) == test_labels)[0].tolist(), 10)
    bad_preds = random.sample(np.where((test_pred >= 0.5) != test_labels)[0].tolist(), 10)

    for name, predictions in {"CorrectPreds": correct_preds, "WrongPreds": bad_preds}.items():
        fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 25))
        for i, img_index in enumerate(predictions):
            row = i // 2
            col = (i % 2) * 2

            img1, img2 = test_image_pairs[img_index]

            axes[row, col].imshow(img1, cmap='gray')
            axes[row, col].axis('off')

            axes[row, col + 1].imshow(img2, cmap='gray')
            axes[row, col + 1].axis('off')

        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(os.getcwd(), f"{name}Test.png"))


if __name__ == "__main__":
    main()
