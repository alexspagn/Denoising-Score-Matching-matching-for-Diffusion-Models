import os
import PIL
from .vision import VisionDataset

class Car(VisionDataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.file_list[idx])
        image = PIL.Image.open(img_name).convert("RGB")

        target = 0

        if self.transform:
            image = self.transform(image)

        return image, target

