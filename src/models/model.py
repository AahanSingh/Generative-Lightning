"""Contains model code for the CycleGAN
"""
import numpy as np
import torch
from torch import nn
import torch.utils.data
import pytorch_lightning as pl
from .losses import discriminator_loss, generator_loss, cycle_loss, identity_loss
from .utils import downsample, upsample


def weights_init(m):
    """Initializes weights with a 0 mean and 0.02 stdev

    Args:
        m (torch.nn.Module): torch module whose weights will be initialized
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.down_stack = nn.ModuleList([
            downsample(in_channels=3, out_channels=64, kernel_size=2, apply_instancenorm=False),
            downsample(in_channels=64, out_channels=128, kernel_size=2),
            downsample(in_channels=128, out_channels=256, kernel_size=2),
            downsample(in_channels=256, out_channels=512, kernel_size=2),
            downsample(in_channels=512, out_channels=512, kernel_size=2),
            downsample(in_channels=512, out_channels=512, kernel_size=2),
            downsample(in_channels=512, out_channels=512, kernel_size=2),
            downsample(in_channels=512, out_channels=512, kernel_size=2, apply_instancenorm=False),
        ])
        self.up_stack = nn.ModuleList([
            upsample(in_channels=512, out_channels=512, kernel_size=2, apply_dropout=True),
            upsample(in_channels=1024, out_channels=512, kernel_size=2, apply_dropout=True),
            upsample(in_channels=1024, out_channels=512, kernel_size=2, apply_dropout=True),
            upsample(in_channels=1024, out_channels=512, kernel_size=2),
            upsample(in_channels=1024, out_channels=256, kernel_size=2),
            upsample(in_channels=512, out_channels=128, kernel_size=2),
            upsample(in_channels=256, out_channels=64, kernel_size=2)
        ])
        self.last = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=2, stride=2), nn.Tanh())

    def forward(self, input):
        # Downsampling through the model
        x = input
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = torch.cat((x, skip), dim=1)
        x = self.last(x)
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            downsample(in_channels=3, out_channels=64, kernel_size=2, apply_instancenorm=False),
            downsample(in_channels=64, out_channels=128, kernel_size=2, apply_instancenorm=False),
            downsample(in_channels=128, out_channels=256, kernel_size=2, apply_instancenorm=False),
            nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, bias=False),
            nn.InstanceNorm2d(num_features=512), nn.LeakyReLU(), nn.ZeroPad2d(1),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1))

    def forward(self, input):
        return self.layers(input)


class CycleGAN(pl.LightningModule):
    """Defines pytorch lightning model for a CycleGAN
    """

    def __init__(
        self,
        lambda_cycle=10,
    ):
        super().__init__()
        self.m_gen = nn.Sequential(Generator())
        self.m_gen.apply(weights_init)
        self.p_gen = nn.Sequential(Generator())
        self.p_gen.apply(weights_init)
        self.m_disc = nn.Sequential(Discriminator())
        self.m_disc.apply(weights_init)
        self.p_disc = nn.Sequential(Discriminator())
        self.p_disc.apply(weights_init)
        self.lambda_cycle = lambda_cycle
        self.gen_loss_fn = generator_loss
        self.cycle_loss_fn = cycle_loss
        self.identity_loss_fn = identity_loss
        self.disc_loss_fn = discriminator_loss

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler
        :return: output - Initialized optimizer and scheduler
        """
        self.monet_generator_optimizer = torch.optim.Adam(self.m_gen.parameters(),
                                                          lr=2e-4,
                                                          betas=(0.5, 0.999))
        self.photo_generator_optimizer = torch.optim.Adam(self.p_gen.parameters(),
                                                          lr=2e-4,
                                                          betas=(0.5, 0.999))

        self.monet_discriminator_optimizer = torch.optim.Adam(self.m_disc.parameters(),
                                                              lr=2e-4,
                                                              betas=(0.5, 0.999))
        self.photo_discriminator_optimizer = torch.optim.Adam(self.p_disc.parameters(),
                                                              lr=2e-4,
                                                              betas=(0.5, 0.999))

        return [
            self.monet_generator_optimizer,
            self.photo_generator_optimizer,
            self.monet_discriminator_optimizer,
            self.photo_discriminator_optimizer,
        ]

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        real_monet, real_photo = train_batch
        if optimizer_idx == 0:
            cycled_photo, cycled_monet, disc_fake_monet, same_monet = self.forward(
                real_monet, real_photo, idx=optimizer_idx)
            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(real_monet, cycled_monet,
                                                  self.lambda_cycle) + self.cycle_loss_fn(
                                                      real_photo, cycled_photo, self.lambda_cycle)
            # evaluates generator loss
            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            # evaluates total generator loss
            total_monet_gen_loss = (
                monet_gen_loss + total_cycle_loss +
                self.identity_loss_fn(real_monet, same_monet, self.lambda_cycle))
            return total_monet_gen_loss

        if optimizer_idx == 1:
            cycled_photo, cycled_monet, disc_fake_photo, same_photo = self.forward(
                real_monet, real_photo, idx=optimizer_idx)
            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(real_monet, cycled_monet,
                                                  self.lambda_cycle) + self.cycle_loss_fn(
                                                      real_photo, cycled_photo, self.lambda_cycle)
            # evaluates generator loss
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)
            # evaluates total generator loss
            total_photo_gen_loss = (
                photo_gen_loss + total_cycle_loss +
                self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle))
            return total_photo_gen_loss

        if optimizer_idx == 2:
            disc_fake_monet, disc_real_monet = self.forward(real_monet,
                                                            real_photo,
                                                            idx=optimizer_idx)
            # evaluates discriminator loss
            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
            return monet_disc_loss

        if optimizer_idx == 3:
            disc_fake_photo, disc_real_photo = self.forward(real_monet,
                                                            real_photo,
                                                            idx=optimizer_idx)
            # evaluates discriminator loss
            photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)
            return photo_disc_loss

    def forward(self, real_monet, real_photo, idx):
        if idx == 0:  # Monet Generator
            # Cycle the Photo
            fake_monet = self.m_gen(real_photo)
            cycled_photo = self.p_gen(fake_monet)
            # monet to photo back to monet
            fake_photo = self.p_gen(real_monet)
            cycled_monet = self.m_gen(fake_photo)
            # discriminator used to check, inputing fake images
            disc_fake_monet = self.m_disc(fake_monet)
            # generating itself
            same_monet = self.m_gen(real_monet)
            return cycled_photo, cycled_monet, disc_fake_monet, same_monet

        if idx == 1:  # Photo Generator
            # Cycle the Photo
            fake_monet = self.m_gen(real_photo)
            cycled_photo = self.p_gen(fake_monet)
            # monet to photo back to monet
            fake_photo = self.p_gen(real_monet)
            cycled_monet = self.m_gen(fake_photo)
            # discriminator used to check, inputing fake images
            disc_fake_photo = self.p_disc(fake_photo)
            # generating itself
            same_photo = self.p_gen(real_photo)
            return cycled_photo, cycled_monet, disc_fake_photo, same_photo

        if idx == 2:  # Monet Discriminator
            # photo to monet back to photo
            fake_monet = self.m_gen(real_photo)
            # discriminator used to check, inputing fake images
            disc_fake_monet = self.m_disc(fake_monet)
            # discriminator used to check, inputing real images
            disc_real_monet = self.m_disc(real_monet)
            return disc_fake_monet, disc_real_monet

        if idx == 3:  # Photo Discriminator
            # photo to monet back to photo
            fake_photo = self.p_gen(real_monet)
            # discriminator used to check, inputing fake images
            disc_fake_photo = self.p_disc(fake_photo)
            # discriminator used to check, inputing real images
            disc_real_photo = self.p_disc(real_photo)
            return disc_fake_photo, disc_real_photo

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs) -> None:
        loss = loss.mean()
        loss.backward()

    def training_step_end(self, losses):
        self.log("train/monet_gen_loss", losses[0].mean(), prog_bar=True, on_step=True)
        self.log("train/photo_gen_loss", losses[1].mean(), prog_bar=True, on_step=True)
        self.log("train/monet_disc_loss", losses[2].mean(), prog_bar=True, on_step=True)
        self.log("train/photo_disc_loss", losses[3].mean(), prog_bar=True, on_step=True)

    """
    def on_train_epoch_end(self, losses) -> None:
        # total_monet_gen_loss, total_photo_gen_loss, monet_disc_loss, photo_disc_loss
        self.log("train/monet_gen_loss", losses[0].mean(), prog_bar=True)
        self.log("train/photo_gen_loss", losses[1].mean(), prog_bar=True)
        self.log("train/monet_disc_loss", losses[2].mean(), prog_bar=True)
        self.log("train/photo_disc_loss", losses[3].mean(), prog_bar=True)
        return
    """