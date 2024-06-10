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
import matplotlib.pyplot as plt


np.random.seed(0)
torch.manual_seed(0)


DATASET_FOLDER = "lfw2"
LEARNING_RATE = 5e-4
BATCH_SIZE = 128
EPOCHS = 200
PATIENCE = 20
LAMBDA = 0.95
L2_REG = 1e-5
RATIO = 0.1


def plot_image_pairs(predictions, test_image_pairs, title):
    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 25))
    for i, img_index in enumerate(predictions):
        col = i // 5
        row = i % 5

        img1, img2 = test_image_pairs[img_index]

        axes[row, col * 2].imshow(img1, cmap='gray')
        axes[row, col * 2].axis('off')

        axes[row, col * 2 + 1].imshow(img2, cmap='gray')
        axes[row, col * 2 + 1].axis('off')

    for col, label in enumerate(["Label 0", "Label 1"]):
        axes[0, col * 2].set_title(label, fontsize=25)
        axes[0, col * 2 + 1].set_title(label, fontsize=25)

    plt.tight_layout()
    plt.suptitle(title, fontsize=30)
    plt.subplots_adjust(top=0.95)
    plt.show()
    plt.savefig(os.path.join(os.getcwd(), f"{title}.png"))


def main():
    ds = PrepareDataset(directory=os.path.join(os.getcwd(), DATASET_FOLDER))
    train_image_pairs, train_labels, validation_image_pairs, validation_labels = ds.load_dataset(
        file_path=os.path.join(os.getcwd(), "pairsDevTrain.txt"), ratio=RATIO
    )
    train_dataset = FacesDataLoader(
        images=train_image_pairs,
        labels=train_labels,
        transform=transforms.Compose([transforms.Grayscale(), transforms.Resize((105, 105)), transforms.ToTensor()])
    )
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    validation_dataset = FacesDataLoader(
        images=validation_image_pairs,
        labels=validation_labels,
        transform=transforms.Compose([transforms.Grayscale(), transforms.Resize((105, 105)), transforms.ToTensor()])
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

    test_image_pairs, test_labels = ds.load_dataset(file_path=os.path.join(os.getcwd(), "pairsDevTest.txt"),
                                                    mode="test")
    test_dataset = FacesDataLoader(
        images=test_image_pairs,
        labels=test_labels,
        transform=transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])
    )
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loss, test_auc, test_accuracy, test_pred = model.evaluate_model(test_dataloader, criterion)
    print(f"Test loss: {test_loss:.3f} | Test Auc: {test_auc:.3f} | Test Accuracy: {test_accuracy:.3f}")

    # Separate correct and incorrect predictions
    correct_indices = np.where((test_pred >= 0.5) == test_labels)[0]
    incorrect_indices = np.where((test_pred >= 0.5) != test_labels)[0]

    # Separate into actual labels 1 and 0
    correct_preds_1 = random.sample([i for i in correct_indices if test_labels[i] == 1], 5)
    correct_preds_0 = random.sample([i for i in correct_indices if test_labels[i] == 0], 5)

    # Separate incorrect predictions into actual labels 1 and 0
    wrong_preds_1 = random.sample([i for i in incorrect_indices if test_labels[i] == 1], 5)
    wrong_preds_0 = random.sample([i for i in incorrect_indices if test_labels[i] == 0], 5)

    # Combine them to get 10 pairs for correct and incorrect
    correct_preds = correct_preds_0 + correct_preds_1
    wrong_preds = wrong_preds_0 + wrong_preds_1

    plot_image_pairs(correct_preds, test_image_pairs, "Correct Predictions")
    plot_image_pairs(wrong_preds, test_image_pairs, "Wrong Predictions")


if __name__ == "__main__":
    main()
