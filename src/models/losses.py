import torch
import torch.utils.data
import torch.nn.functional as F


def discriminator_loss(real, generated):
    real_loss = F.binary_cross_entropy_with_logits(
        input=real, target=torch.ones_like(real), reduction=None
    )
    generated_loss = F.binary_cross_entropy_with_logits(
        input=generated, target=torch.zeros_like(generated), reduction=None
    )
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5


def generator_loss(generated):
    return F.binary_cross_entropy_with_logits(
        inputs=generated, target=torch.ones_like(generated), reduction=None
    )


def cycle_loss(real_image, cycled_image, LAMBDA):
    loss1 = torch.mean(torch.abs(real_image - cycled_image))
    return LAMBDA * loss1


def identity_loss(real_image, same_image, LAMBDA):
    loss = torch.mean(torch.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss