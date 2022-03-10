import os
from argparse import ArgumentParser
from asyncio.log import logger
import munch
import yaml
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
import albumentations as A
from albumentations.pytorch import ToTensorV2
from lightning_gans.models.discriminators import Discriminator

from lightning_gans.models.generators import UNETGenerator, WideResnetEncoderDecoder, WideResnetUNET, CustomUNET
from lightning_gans.models.cycle_gan import CycleGAN
from lightning_gans.data.dataloader import MonetDataset

augmentations = {
    "horizontalflip": A.HorizontalFlip(p=0.5),
    "randomcrop": A.RandomResizedCrop(height=256, width=256, p=0.5),
}
generators = {
    "UNET": UNETGenerator,
    "Resnet": WideResnetEncoderDecoder,
    "WideResnetUNET": WideResnetUNET,
    "CustomUNET": CustomUNET,
}


def main(args):
    with open(args.config_path, "r") as f:
        config = munch.munchify(yaml.safe_load(f))
    os.makedirs(
        "{}/{}/{}".format(
            config.savedir,
            config.experiment_name,
            config.run_name,
        ),
        exist_ok=True,
    )

    if not config.id:
        config.id = wandb.util.generate_id()
    with open(
            "{}/{}/{}/config.yaml".format(
                config.savedir,
                config.experiment_name,
                config.run_name,
            ),
            "w",
    ) as f:
        yaml.dump(munch.unmunchify(config), f)
    # Root directory for dataset
    dataroot = config.data_path
    # Number of workers for dataloader
    workers = 12
    # Batch size during training
    batch_size = config.batch_size
    # Spatial size of training images. All images will be resized to this
    #   size using a transformer.
    image_size = 256
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    train_transforms = [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    if config.augmentations is not None:
        tmp = [augmentations[i] for i in config.augmentations]
        tmp.extend(train_transforms)
        train_transforms = tmp
    train_transforms = A.Compose(train_transforms)
    train_dataset = MonetDataset(dataroot=dataroot,
                                 transforms=train_transforms,
                                 shuffle=config.dataset.shuffle,
                                 complete_dataset=config.dataset.complete)
    # Create the dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=workers)
    # Create the generator

    model = CycleGAN(generator=generators[config.generator],
                     discriminator=Discriminator,
                     l=config.l,
                     k=config.k)
    wandb_logger = WandbLogger(project="Monet CycleGAN",
                               log_model=True,
                               save_dir="{}/{}/{}/".format(
                                   config.savedir,
                                   config.experiment_name,
                                   config.run_name,
                               ),
                               name="{}-{}".format(config.experiment_name, config.run_name),
                               id=config.id,
                               config=munch.unmunchify(config))
    wandb_logger.watch(model)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="train/monet_gen_loss",
        mode="min",
        filename="model-monet_generator_loss{train/monet_gen_loss:.2f}",
        auto_insert_metric_name=False,
        dirpath="{}/{}/{}/models/".format(
            config.savedir,
            config.experiment_name,
            config.run_name,
        ),
        every_n_train_steps=1)
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=config.epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_dataloader=train_dataloader, ckpt_path=config.ckpt_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config-path",
        required=True,
        metavar="FILE",
        help="Path to the config",
    )
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    args = parser.parse_args()
    main(args)