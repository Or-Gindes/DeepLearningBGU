import os
from PIL import Image


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

    def load_dataset(self, file_path: str):  # -> tuple[list[tuple[Image, Image]], list[int]]:
        """
        reads a txt file with predefined pairs and creates a dataset as a dataloader object.
        :param file_path: string path to txt file with the desired pairs in the dataset
        :return: DataLoader
        """
        image_pairs = []
        labels = []
        with open(file_path, 'r') as f:
            rows = f.readlines()
            num_samples = len(rows)
            i = 1
            while i < num_samples:
                if i == 1900:
                    pass
                row_segments = rows[i].strip().split()
                if len(row_segments) == 3:
                    person = row_segments[0]

                    image1 = Image.open(os.path.join(self.directory,
                                                     person,
                                                     self.image_dict[person][int(row_segments[1]) - 1]))
                    image2 = Image.open(os.path.join(self.directory,
                                                     person,
                                                     self.image_dict[person][int(row_segments[2]) - 1]))
                    labels.append(1)
                else:
                    person1 = row_segments[0]
                    image1 = Image.open(os.path.join(self.directory,
                                                     person1,
                                                     self.image_dict[person1][int(row_segments[1]) - 1]))
                    person2 = row_segments[2]
                    image2 = Image.open(os.path.join(self.directory,
                                                     person2,
                                                     self.image_dict[person2][int(row_segments[3]) - 1]))
                    labels.append(0)

                image_pairs.append((image1, image2))
                i += 1
        return image_pairs, labels
