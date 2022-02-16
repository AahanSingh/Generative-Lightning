import torch
import torch.utils.data
import pytorch_lightning as pl

from .losses import *


class CycleGAN(pl.LightningModule):
    def __init__(
        self,
        monet_generator,
        photo_generator,
        monet_discriminator,
        photo_discriminator,
        lambda_cycle=10,
    ):
        super().__init__()
        self.m_gen = monet_generator
        self.p_gen = photo_generator
        self.m_disc = monet_discriminator
        self.p_disc = photo_discriminator
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
        self.monet_generator_optimizer = torch.optim.Adam(
            self.m_gen.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )
        self.photo_generator_optimizer = torch.optim.Adam(
            self.p_gen.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )

        self.monet_discriminator_optimizer = torch.optim.Adam(
            self.m_disc.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )
        self.photo_discriminator_optimizer = torch.optim.Adam(
            self.p_disc.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )

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
                real_monet, real_photo, idx=optimizer_idx
            )
            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(
                real_monet, cycled_monet, self.lambda_cycle
            ) + self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle)
            # evaluates generator loss
            monet_gen_loss = self.gen_loss_fn(disc_fake_monet)
            # evaluates total generator loss
            total_monet_gen_loss = (
                monet_gen_loss
                + total_cycle_loss
                + self.identity_loss_fn(real_monet, same_monet, self.lambda_cycle)
            )
            return total_monet_gen_loss

        if optimizer_idx == 1:
            cycled_photo, cycled_monet, disc_fake_photo, same_photo = self.forward(
                real_monet, real_photo, idx=optimizer_idx
            )
            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(
                real_monet, cycled_monet, self.lambda_cycle
            ) + self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle)
            # evaluates generator loss
            photo_gen_loss = self.gen_loss_fn(disc_fake_photo)
            # evaluates total generator loss
            total_photo_gen_loss = (
                photo_gen_loss
                + total_cycle_loss
                + self.identity_loss_fn(real_photo, same_photo, self.lambda_cycle)
            )
            return total_photo_gen_loss

        if optimizer_idx == 2:
            disc_fake_monet, disc_real_monet = self.forward(
                real_monet, real_photo, idx=optimizer_idx
            )
            # evaluates discriminator loss
            monet_disc_loss = self.disc_loss_fn(disc_real_monet, disc_fake_monet)
            return monet_disc_loss

        if optimizer_idx == 3:
            disc_fake_photo, disc_real_photo = self.forward(
                real_monet, real_photo, idx=optimizer_idx
            )
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
        loss.backward()
