from prepareDataset import *
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler 
from prepareDataset import *
from torchDataloader import *
from siameseNetwork import *
from earlyStopping import EarlyStopping

device = ("cuda" if torch.cuda.is_available() else "cpu")

def main():
    ds = PrepareDataset(directory='./lfw2')
    train_image_pairs, train_labels, validation_image_pairs, validation_labels = ds.load_dataset(file_path=r'./pairsDevTrain.txt', mode='train')

    train_dataset = FacesDataLoader(images=train_image_pairs,
                                    labels=train_labels,
                                    transform=transforms.Compose([transforms.ToTensor()]))

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    validation_dataset = FacesDataLoader(images=validation_image_pairs,
                                         labels=validation_labels,
                                         transform=transforms.Compose([transforms.ToTensor()]))

    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True)

    test_image_pairs, test_labels = ds.load_dataset(file_path=r'./pairsDevTest.txt', mode='test')

    test_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=True)

    model = SiameseNetwork().to(device)
    lr = 0.05
    optimizer = Adam(model.parameters(), lr=lr)
    lambda_ = lambda epoch: 0.99
    scheduler = lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda_)
    early_stopping = EarlyStopping(patience=20)
    model.train_model(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        epoch=200, 
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping
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
