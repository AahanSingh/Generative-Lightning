from argparse import ArgumentParser
from asyncio.log import logger
import random
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
#from src.visualization.callbacks import WandbImageCallback

from lightning_gans.models.generators import UNETGenerator, WideResnetEncoderDecoder
from lightning_gans.models.cycle_gan import CycleGAN
from lightning_gans.data.dataloader import MonetDataset

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
    batch_size = args.bs
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 256
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    train_dataset = MonetDataset(dataroot=dataroot,
                                 transforms=transforms.Compose([
                                     transforms.Resize(image_size),
                                     transforms.CenterCrop(image_size),
                                     transforms.Normalize(127.5, 127.5),
                                 ]))
    # Create the dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=workers)
    val_dataset = MonetDataset(dataroot=dataroot,
                               transforms=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.Normalize(127.5, 127.5),
                               ]),
                               shuffle_photos=True,
                               val=True,
                               val_size=20)
    # Create the dataloader
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=batch_size,
                                                 num_workers=workers)
    # Create the generator
    generators = {"UNET": UNETGenerator, "Resnet": WideResnetEncoderDecoder}
    model = CycleGAN(generator=generators[args.gen], l=args.l, k=args.k)
    wandb_logger = WandbLogger(project="Monet CycleGAN", log_model="all")
    wandb_logger.watch(model)
    trainer = pl.Trainer.from_argparse_args(
        args, logger=wandb_logger)  #, callbacks=[WandbImageCallback(val_dataloader)])

    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data-path", metavar="FILE", default=None, required=True)
    parser.add_argument("--bs", default=128, type=int)
    parser.add_argument("--gen", required=True, type=str)
    parser.add_argument("--l", required=True, type=int)
    parser.add_argument("--k", required=True, type=int)
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    args = parser.parse_args()
    main(args)