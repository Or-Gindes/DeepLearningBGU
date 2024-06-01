import pandas
from dataset import PrepareDataset
import os
import numpy as np
from PIL import Image

def main():
    # print(os.getcwd())
    # dataset.Dataset(r'./pairsDevTrain.txt', r'./lfw2', train=True)
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
