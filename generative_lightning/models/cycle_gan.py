"""Contains model code for the CycleGAN
"""
import itertools
import torch
from torch import nn
import torch.utils.data
import pytorch_lightning as pl
from .losses import discriminator_loss, generator_loss, cycle_loss, identity_loss


def weights_init(m):
    """Initializes weights with a 0 mean and 0.02 stdev

    Args:
        m (torch.nn.Module): torch module whose weights will be initialized
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print(classname)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('InstanceNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class CycleGAN(pl.LightningModule):
    """Defines pytorch lightning model for a CycleGAN
    """

    def __init__(
        self,
        generator,
        discriminator,
        lambda_cycle=10,
        separate_optimizers=True,
        **kwargs,
    ):
        super().__init__()
        self.m_gen = nn.Sequential(generator(**kwargs))
        self.m_gen.apply(weights_init)
        self.p_gen = nn.Sequential(generator(**kwargs))
        self.p_gen.apply(weights_init)
        self.m_disc = nn.Sequential(discriminator(**kwargs))
        self.m_disc.apply(weights_init)
        self.p_disc = nn.Sequential(discriminator(**kwargs))
        self.p_disc.apply(weights_init)
        self.lambda_cycle = lambda_cycle
        self.gen_loss_fn = generator_loss
        self.cycle_loss_fn = cycle_loss
        self.identity_loss_fn = identity_loss
        self.disc_loss_fn = discriminator_loss
        self.separate_optimizers = separate_optimizers

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler
        :return: output - Initialized optimizer and scheduler
        """
        if self.separate_optimizers:
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
        else:
            self.generator_optimizer = torch.optim.Adam(itertools.chain(
                self.m_gen.parameters(), self.p_gen.parameters()),
                                                        lr=2e-4,
                                                        betas=(0.5, 0.999))
            self.discriminator_optimizer = torch.optim.Adam(itertools.chain(
                self.m_disc.parameters(), self.p_disc.parameters()),
                                                            lr=2e-4,
                                                            betas=(0.5, 0.999))
            return [self.generator_optimizer, self.discriminator_optimizer]

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        real_monet, real_photo = train_batch
        if self.separate_optimizers:
            if optimizer_idx == 0:
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
                # evaluates total cycle consistency loss
                total_cycle_loss = self.cycle_loss_fn(
                    real_monet, cycled_monet, self.lambda_cycle) + self.cycle_loss_fn(
                        real_photo, cycled_photo, self.lambda_cycle)
                # evaluates generator loss
                monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
                # evaluates total generator loss
                total_monet_gen_loss = (
                    monet_gen_loss + total_cycle_loss +
                    self.identity_loss_fn(real_monet, same_monet, self.lambda_cycle))
                self.log("train/monet_gen_loss", total_monet_gen_loss.item(), prog_bar=True)
                return total_monet_gen_loss

            if optimizer_idx == 1:
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
                # evaluates total cycle consistency loss
                total_cycle_loss = self.cycle_loss_fn(
                    real_monet, cycled_monet, self.lambda_cycle) + self.cycle_loss_fn(
                        real_photo, cycled_photo, self.lambda_cycle)
                # evaluates generator loss
                photo_gen_loss = self.gen_loss_fn(disc_fake_photo)
                # evaluates total generator loss
                total_photo_gen_loss = (
                    photo_gen_loss + total_cycle_loss +
                    self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle))
                self.log("train/photo_gen_loss", total_photo_gen_loss, prog_bar=True)
                return total_photo_gen_loss

            if optimizer_idx == 2:
                # photo to monet back to photo
                fake_monet = self.m_gen(real_photo)
                # discriminator used to check, inputing fake images
                disc_fake_monet = self.m_disc(fake_monet)
                # discriminator used to check, inputing real images
                disc_real_monet = self.m_disc(real_monet)
                # evaluates discriminator loss
                monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
                self.log("train/monet_disc_loss", monet_disc_loss, prog_bar=True)
                return monet_disc_loss

            if optimizer_idx == 3:
                # photo to monet back to photo
                fake_photo = self.p_gen(real_monet)
                # discriminator used to check, inputing fake images
                disc_fake_photo = self.p_disc(fake_photo)
                # discriminator used to check, inputing real images
                disc_real_photo = self.p_disc(real_photo)
                # evaluates discriminator loss
                photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)
                self.log("train/photo_disc_loss", photo_disc_loss, prog_bar=True)
                return photo_disc_loss
        else:
            if optimizer_idx == 0:
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
                # discriminator used to check, inputing fake images
                disc_fake_photo = self.p_disc(fake_photo)
                # generating itself
                same_photo = self.p_gen(real_photo)
                # evaluates total cycle consistency loss
                total_cycle_loss = self.cycle_loss_fn(
                    real_monet, cycled_monet, self.lambda_cycle) + self.cycle_loss_fn(
                        real_photo, cycled_photo, self.lambda_cycle)
                # evaluates generator loss
                monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
                # evaluates total generator loss
                total_monet_gen_loss = (
                    monet_gen_loss + total_cycle_loss +
                    self.identity_loss_fn(real_monet, same_monet, self.lambda_cycle))
                # evaluates generator loss
                photo_gen_loss = self.gen_loss_fn(disc_fake_photo)
                # evaluates total generator loss
                total_photo_gen_loss = (
                    photo_gen_loss + total_cycle_loss +
                    self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle))
                self.log("train/total_gen_loss",
                         total_monet_gen_loss.item() + total_photo_gen_loss.item(),
                         prog_bar=True)
                return total_monet_gen_loss + total_photo_gen_loss

            if optimizer_idx == 1:
                # photo to monet back to photo
                fake_monet = self.m_gen(real_photo)
                # discriminator used to check, inputing fake images
                disc_fake_monet = self.m_disc(fake_monet)
                # discriminator used to check, inputing real images
                disc_real_monet = self.m_disc(real_monet)

                # photo to monet back to photo
                fake_photo = self.p_gen(real_monet)
                # discriminator used to check, inputing fake images
                disc_fake_photo = self.p_disc(fake_photo)
                # discriminator used to check, inputing real images
                disc_real_photo = self.p_disc(real_photo)
                # evaluates discriminator loss
                monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
                # evaluates discriminator loss
                photo_disc_loss = self.disc_loss_fn(disc_real_photo, disc_fake_photo)
                self.log("train/total_disc_loss",
                         monet_disc_loss.item() + photo_disc_loss.item(),
                         prog_bar=True)
                return monet_disc_loss + photo_disc_loss

    def forward(self, image, generate_monet=True):
        if generate_monet:
            return self.m_gen(image)
        else:
            return self.p_gen(image)

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs) -> None:
        loss.backward()