import os
from typing import Set, Any, Tuple, List
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class PrepareDataset:
    def __init__(self, directory: str):
        self.directory = directory
        self.image_dict = {}
        self.scan_dataset()

    def scan_dataset(self):
        """
        scans through all available files, creates a full dataset to be used to create a validation dataset
        """
        # Iterate through each directory in the base folder
        for person_name in os.listdir(self.directory):
            person_name_folder = os.path.join(self.directory, person_name)

            if os.path.isdir(person_name_folder):
                images = []

                # Iterate through each file in the person's folder
                for file_name in os.listdir(person_name_folder):
                    # Check if the file name matches the pattern "<person_name>_<image number in 4d format>"
                    if file_name.endswith('.jpg'):
                        images.append(file_name)

                # Add the list of image files to the dictionary
                self.image_dict[person_name] = images

    def load_dataset(self, file_path: str, mode: str = 'train') -> tuple[list[tuple[Any, Any]], list[int]] | tuple[
        list[Any], list[int], list[Any], list[int]]:
        """
        reads a txt file with predefined pairs and creates:
            mode = 'train' : train and validation datasets.
            mode = 'test' : test dataset.
        :param file_path: string path to txt file with the desired pairs in the dataset
        :param mode: 'train', 'test'. default = 'train',
        :return: train_image_pairs, train_labels, validation_image_pairs, validation_labels | test_image_pairs, test_labels
        """
        if mode == 'test':
            test_image_pairs = []
            test_labels = []
            with open(file_path, 'r') as f:
                rows = f.readlines()
                num_samples = len(rows)
                i = 1
                while i < num_samples:
                    row_segments = rows[i].strip().split()
                    if len(row_segments) == 3:
                        person = row_segments[0]
                        image1_path = os.path.join(self.directory,
                                                   person,
                                                   self.image_dict[person][int(row_segments[1]) - 1])
                        image2_path = os.path.join(self.directory,
                                                   person,
                                                   self.image_dict[person][int(row_segments[2]) - 1])
                        test_labels.append(1)
                    else:
                        person1 = row_segments[0]
                        image1_path = os.path.join(self.directory,
                                                   person1,
                                                   self.image_dict[person1][int(row_segments[1]) - 1])
                        person2 = row_segments[2]
                        image2_path = os.path.join(self.directory,
                                                   person2,
                                                   self.image_dict[person2][int(row_segments[3]) - 1])
                        test_labels.append(0)

                    with Image.open(image1_path) as img1, Image.open(image2_path) as img2:
                        test_image_pairs.append((img1.copy(), img2.copy()))
                    i += 1

                    return test_image_pairs, test_labels

        elif mode == 'train':
            people_in_validation = self.train_validation_split(file_path=file_path, ratio=0.2)
            train_image_pairs = []
            train_labels = []
            validation_image_pairs = []
            validation_labels = []
            with open(file_path, 'r') as f:
                rows = f.readlines()
                num_samples = len(rows)
                i = 1
                while i < num_samples:
                    row_segments = rows[i].strip().split()
                    if len(row_segments) == 3:
                        person = row_segments[0]
                        image1_path = os.path.join(self.directory,
                                                   person,
                                                   self.image_dict[person][int(row_segments[1]) - 1])
                        image2_path = os.path.join(self.directory,
                                                   person,
                                                   self.image_dict[person][int(row_segments[2]) - 1])
                        if person in people_in_validation:
                            image_pairs = validation_image_pairs
                            labels = validation_labels
                        else:
                            image_pairs = train_image_pairs
                            labels = train_labels
                        labels.append(1)
                    else:
                        person1 = row_segments[0]
                        image1_path = os.path.join(self.directory,
                                                   person1,
                                                   self.image_dict[person1][int(row_segments[1]) - 1])
                        person2 = row_segments[2]
                        image2_path = os.path.join(self.directory,
                                                   person2,
                                                   self.image_dict[person2][int(row_segments[3]) - 1])
                        if person1 in people_in_validation or person2 in people_in_validation:
                            image_pairs = validation_image_pairs
                            labels = validation_labels
                        else:
                            image_pairs = train_image_pairs
                            labels = train_labels
                        labels.append(0)
                    with Image.open(image1_path) as img1, Image.open(image2_path) as img2:
                        image_pairs.append((img1.copy(), img2.copy()))
                    i += 1
            return train_image_pairs, train_labels, validation_image_pairs, validation_labels
        else:
            raise ValueError(f'Invalid "mode" variable, expected ["train", "test"], received "{mode}"')

    def train_validation_split(self, file_path: str, ratio: float = 0.2) -> set[object]:
        """
        returns train set and validation set of unique people for one-shot learning.
        This division is set up so that no subject from the validation set is included in the train set.
        :param file_path: string path to train txt file with the desired pairs in the dataset
        :param ratio: initial size ratio of validation set. Due to the need of unique unseen individuals, this may grow.
        """
        with open(file_path, 'r') as f:
            rows = f.readlines()
            num_samples = len(rows)
            i = 1
            person_a = []
            person_b = []
            while i < num_samples:
                row_segments = rows[i].strip().split()
                if len(row_segments) == 3:
                    person_a.append(row_segments[0])
                    person_b.append(row_segments[0])
                else:
                    person_a.append(row_segments[0])
                    person_b.append(row_segments[2])

                i += 1
        person_a, person_b = person_a + person_b, person_b + person_a
        assert len(person_a) == len(person_b)
        pairs = pd.DataFrame({'A': person_a, 'B': person_b})
        pairs.drop_duplicates(inplace=True)
        assert len(pairs) % 2 == 0
        pairs = pairs.iloc[:int(len(pairs) / 2)]
        train_pairs, validation_pairs = train_test_split(pairs, test_size=ratio, random_state=1)
        people_in_validation = set(np.unique(validation_pairs['A'])).union(set(np.unique(validation_pairs['B'])))

        return people_in_validation
