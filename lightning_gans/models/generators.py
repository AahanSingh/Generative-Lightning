"""Contains model code for the CycleGAN
"""
import torch
from torch import nn
import torch.utils.data
from .utils import downsample, upsample


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
