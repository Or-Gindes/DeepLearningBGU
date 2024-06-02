from torch.utils.data import Dataset


class FacesDataLoader(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return int(len(self.images))

    def __getitem__(self, idx):
        x1, x2 = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)

        return x1, x2, label
