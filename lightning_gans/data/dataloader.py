"""
This modules contains all of the pytorch datasets used in the project.
"""
import itertools
import glob
import random
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MonetDataset(Dataset):
    """Pytorch Dataset Subclass for the Monet Dataset
    """

    def __init__(self, dataroot, transforms=None, shuffle=True, complete_dataset=True):
        super().__init__()
        self.monets = glob.glob(dataroot + "monet_jpg/*.jpg")
        self.photos = glob.glob(dataroot + "photo_jpg/*.jpg")
        if shuffle:
            random.shuffle(self.photos)
            random.shuffle(self.monets)
        self.transforms = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=0, std=1, max_pixel_value=255),
            ToTensorV2(),
        ])
        if transforms is not None:
            self.transforms = transforms
        self.aligned_images = None
        if complete_dataset:
            self.aligned_images = list(zip(itertools.cycle(self.monets), self.photos))
        else:
            self.aligned_images = list(zip(self.monets, self.photos))

    def _read_image(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, idx):
        monet = self._read_image(self.aligned_images[idx][0])
        photo = self._read_image(self.aligned_images[idx][1])
        monet = self.transforms(image=monet)["image"].float()
        photo = self.transforms(image=photo)["image"].float()
        return monet, photo

    def __len__(self):
        return len(self.aligned_images)
