"""Contains model code for the CycleGAN
"""
import math
import torch
from torch import nn
import torch.utils.data
from .utils import downsample, upsample
from .resnet import WideResnetEncoder, WideResnetDecoder, conv_downsample, conv_upsample, ResidualLayer


class UNETGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        self.down_stack = nn.ModuleList([
            downsample(in_channels=3, out_channels=64, kernel_size=3, apply_instancenorm=False),
            downsample(in_channels=64, out_channels=128, kernel_size=3),
            downsample(in_channels=128, out_channels=256, kernel_size=3),
            downsample(in_channels=256, out_channels=512, kernel_size=3),
            downsample(in_channels=512, out_channels=512, kernel_size=3),
            downsample(in_channels=512, out_channels=512, kernel_size=3),
            downsample(in_channels=512, out_channels=512, kernel_size=3),
            downsample(in_channels=512, out_channels=512, kernel_size=3, apply_instancenorm=False),
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
            print(x.shape)
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            print(x.shape, skip.shape)
            x = torch.cat((x, skip), dim=1)
        x = self.last(x)
        return x


class WideResnetEncoderDecoder(torch.nn.Module):

    def __init__(self, l=4, k=4, input_channels=3, image_size=256):
        super().__init__()
        downsample_channels = int(input_channels * 2**(math.log2(image_size) - 1))
        self.encoder = WideResnetEncoder(l=l,
                                         k=k,
                                         input_channels=input_channels,
                                         image_size=image_size)
        self.decoder = WideResnetDecoder(l=l,
                                         k=k,
                                         input_channels=downsample_channels,
                                         image_size=image_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class WideResnetUNET(torch.nn.Module):

    def __init__(self, l=2, k=1, input_channels=3, image_size=256):
        super().__init__()

        self.num_encoder_decoder_layers = int(math.log2(image_size))
        self.res_depth = l
        self.res_width_factor = k
        self.input_channels = input_channels
        self.downsample_filters = []
        self.encoder = self.get_encoder()
        self.decoder = self.get_decoder()
        self.last_conv = conv_upsample(ks=3,
                                       st=2,
                                       in_c=self.downsample_filters[-1] * 2,
                                       out_c=self.input_channels,
                                       activation="tanh")

    def get_encoder(self):
        layers = []
        in_ch = self.input_channels
        f = 2
        for _ in range(self.num_encoder_decoder_layers - 1):
            layer = ResidualLayer(l=self.res_depth,
                                  k=self.res_width_factor,
                                  in_channels=in_ch,
                                  res_channels=f,
                                  kernel_size=3)
            f *= 2
            downsample = conv_downsample(ks=3, st=2, in_c=in_ch, out_c=f, activation="relu")
            layers.append(torch.nn.Sequential(layer, downsample))
            in_ch = f
            self.downsample_filters.append(f)
        self.downsample_filters = list(reversed(self.downsample_filters))
        return torch.nn.ModuleList(layers)

    def get_decoder(self):
        layers = []
        for i in range(1, len(self.downsample_filters)):
            in_ch = self.downsample_filters[i - 1]
            if i != 1:
                in_ch *= 2
            layer = ResidualLayer(l=self.res_depth,
                                  k=self.res_width_factor,
                                  in_channels=in_ch,
                                  res_channels=self.downsample_filters[i - 1],
                                  kernel_size=3)
            upsampling = conv_upsample(ks=3,
                                       st=2,
                                       in_c=in_ch,
                                       out_c=self.downsample_filters[i],
                                       activation="relu")
            layers.append(torch.nn.Sequential(layer, upsampling))

        return torch.nn.ModuleList(layers)

    def forward(self, x):
        d_feats = []
        for l in self.encoder:
            x = l(x)
            d_feats.append(x)
        d_feats.reverse()
        d_feats.pop(0)
        for i, l in enumerate(self.decoder):
            x = l(x)
            x = torch.cat((x, d_feats[i]), 1)
        x = self.last_conv(x)
        return x