import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from prepareDataset import PrepareDataset
from torchDataloader import FacesDataLoader
from siameseNetwork import SiameseNetwork


DATASET_FOLDER = "lfw2"


def main():
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    ds = PrepareDataset(directory=os.path.join(os.getcwd(), DATASET_FOLDER))
    train_image_pairs, train_labels = ds.load_dataset(file_path=os.path.join(os.getcwd(), "pairsDevTrain.txt"))

    train_dataset = FacesDataLoader(
        images=train_image_pairs,
        labels=train_labels,
        transform=transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])
    )

    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)

    validation_image_pairs, validation_labels = ds.load_dataset(file_path=os.path.join(os.getcwd(), "pairsDevTest.txt"))

    validation_dataset = FacesDataLoader(
        images=validation_image_pairs,
        labels=validation_labels,
        transform=transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])
    )

    validation_dataloader = DataLoader(validation_dataset, batch_size=5, shuffle=True)

    model = SiameseNetwork().to(device)
    model.train_model(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        epoch=5,
        learning_rate=.01
    )
    pass

    # transform = transforms.Compose([transforms.Resize((105, 105)), transforms.ToTensor()])
    # folder_dataset = datasets.ImageFolder(root='path_to_your_data')
    # siamese_dataset = SiameseDataset(imageFolderDataset=folder_dataset, transform=transform)
    # train_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=8, batch_size=32)
    #
    # model = SiameseNetwork().to(device)
    # criterion = ContrastiveLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #
    # train(model, train_dataloader, criterion, optimizer


if __name__ == "__main__":
    main()
