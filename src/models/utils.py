"""This module contains helper functions for creation of the UNET
"""
from torch import nn as nn


def downsample(in_channels, out_channels, kernel_size, apply_instancenorm=True):
    """Creates a downsample layer

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the kernel
        apply_instancenorm (bool, optional): If true applies instance norm 
                                            at the outputs of the layer. Defaults to True.

    Returns:
        torch.nn.Sequential: Returns a Sequential layer made up of 
                            a Conv2d, InstanceNorm and LeakyRELU
    """
    layer = nn.Sequential()
    layer.add_module(
        "Conv",
        nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size,
                  stride=2,
                  padding=0,
                  bias=False))
    if apply_instancenorm:
        layer.add_module("InstanceNorm", nn.InstanceNorm2d(num_features=out_channels))
    layer.add_module("LeakyRelu", nn.LeakyReLU())
    return layer


def upsample(in_channels, out_channels, kernel_size, apply_dropout=False):
    """Creates an upsample layer

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the kernel
        apply_dropout (bool, optional):If true applies dropout. Defaults to False.

    Returns:
        torch.nn.Sequential: Returns a Sequential layer made up of a 
                            Transposed Conv2d, InstanceNorm, Dropout and RELU
    """
    layer = nn.Sequential()
    layer.add_module(
        "Conv",
        nn.ConvTranspose2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=2,
                           padding=0,
                           bias=False))
    layer.add_module("InstanceNorm", nn.InstanceNorm2d(num_features=out_channels))
    if apply_dropout:
        layer.add_module("Dropout", nn.Dropout2d(p=0.5))
    layer.add_module("Relu", nn.ReLU())
    return layer