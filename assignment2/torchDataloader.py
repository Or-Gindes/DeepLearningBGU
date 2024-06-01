import torch
from torch.utils.data import Dataset, DataLoader


class FacesDataLoader(Dataset):
    def __int__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return int(len(self.images))

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
