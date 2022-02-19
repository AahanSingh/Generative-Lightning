"""
This modules contains all of the pytorch datasets used in the project.
"""
import glob
import random
import torch
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.io


class MonetDataset(Dataset):
    """Pytorch Dataset Subclass for the Monet Dataset
    """

    def __init__(self, dataroot, transforms=None, shuffle_photos=False):
        super().__init__()
        self.monets = glob.glob(dataroot + "monet_jpg/*.jpg")
        photos = glob.glob(dataroot + "photo_jpg/*.jpg")
        if shuffle_photos:
            random.shuffle(photos)
        self.photos = photos[:len(self.monets)]
        self.transforms = transforms

    def __getitem__(self, idx):
        monet = torchvision.io.read_image(self.monets[idx]).float()
        photo = torchvision.io.read_image(self.photos[idx]).float()
        if self.transforms:
            monet = self.transforms(monet)
            photo = self.transforms(photo)
        return monet, photo

    def __len__(self):
        return len(self.monets)
