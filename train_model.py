from argparse import ArgumentParser
import random
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import pytorch_lightning as pl
from src.models.model import CycleGAN
from src.data.dataloader import MonetDataset

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def main(args):
    # Root directory for dataset
    dataroot = args.data_path
    # Number of workers for dataloader
    workers = 12
    # Batch size during training
    batch_size = 12
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 256
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = MonetDataset(dataroot=dataroot,
                           transforms=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.Normalize(127.5, 127.5),
                           ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers)
    # Create the generator
    model = CycleGAN()
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data-path", metavar="FILE", default=None, required=True)
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    args = parser.parse_args()
    main(args)