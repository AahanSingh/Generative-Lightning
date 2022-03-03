"""
This modules contains all of the pytorch datasets used in the project.
"""
import itertools
import glob
import random
from torch.utils.data import Dataset
import torchvision.io


class MonetDataset(Dataset):
    """Pytorch Dataset Subclass for the Monet Dataset
    """

    def __init__(self, dataroot, transforms=None, shuffle=True):
        super().__init__()
        self.monets = glob.glob(dataroot + "monet_jpg/*.jpg")
        self.photos = glob.glob(dataroot + "photo_jpg/*.jpg")
        if shuffle:
            random.shuffle(self.photos)
            random.shuffle(self.monets)
        self.transforms = transforms
        self.aligned_images = list(zip(itertools.cycle(self.monets), self.photos))

    def __getitem__(self, idx):
        monet = torchvision.io.read_image(self.aligned_images[idx][0]).float()
        photo = torchvision.io.read_image(self.aligned_images[idx][1]).float()
        if self.transforms:
            monet = self.transforms(monet)
            photo = self.transforms(photo)
        return monet, photo

    def __len__(self):
        return len(self.aligned_images)
