"""Contains model code for the CycleGAN
"""
from torch import nn
from .utils import downsample


class Discriminator(nn.Module):

    def __init__(self, **kwargs):
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
