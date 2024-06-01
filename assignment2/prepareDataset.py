import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import random
from PIL import Image


class PrepareDataset:
    def __init__(self, directory: str):
        self.directory = directory
        self.image_dict = {}
        self.transform = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor()]
        )
        self.scan_dataset()

    def scan_dataset(self):
        # Iterate through each directory in the base folder
        for person_name in os.listdir(self.directory):
            person_name_folder = os.path.join(self.directory, person_name)

            #
            if os.path.isdir(person_name_folder):
                images = []

                # Iterate through each file in the person's folder
                for file_name in os.listdir(person_name_folder):
                    # Check if the file name matches the pattern "<person_name>_<image number in 4d format>"
                    if file_name.endswith('.jpg'):
                        images.append(file_name)

                # Add the list of image files to the dictionary
                self.image_dict[person_name] = images

    def load_dataset(self, file):
        image_pairs = []
        labels = []
        with open(file, 'r') as f:
            rows = f.readlines()
            num_samples = int(rows[0].strip())
            i = 1
            while i < num_samples:
                row_segments = rows[i].strip().split()
                if len(row_segments) == 3:
                    person = row_segments[0]
                    image1 = Image.open(self.image_dict[person][int(row_segments[1])-1])
                    image2 = Image.open(self.image_dict[person][int(row_segments[2])-1])
                    labels.append(1)
                else:
                    person1 = row_segments[0]
                    image1 = Image.open(self.image_dict[person1][int(row_segments[1]) - 1])
                    person2 = row_segments[2]
                    image2 = Image.open(self.image_dict[person2][int(row_segments[3]) - 1])
                    labels.append(0)

                image_pairs.append([image1, image2])
