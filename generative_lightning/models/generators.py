"""Contains model code for the CycleGAN
"""
import math
import torch
from torch import nn
import torch.utils.data
from .utils import downsample, upsample
from .resnet import WideResnetEncoder, WideResnetDecoder, conv_downsample, conv_upsample, ResidualLayer


class UNETGenerator(nn.Module):

    def __init__(self, **kwargs):
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


class CustomUNET(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.down_stack = nn.ModuleList([
            conv_downsample(ks=5, st=2, in_c=3, out_c=64, activation="leaky_relu", norm=None),
            conv_downsample(ks=5,
                            st=2,
                            in_c=64,
                            out_c=128,
                            activation="leaky_relu",
                            norm="instance"),
            conv_downsample(ks=5,
                            st=2,
                            in_c=128,
                            out_c=256,
                            activation="leaky_relu",
                            norm="instance"),
            conv_downsample(ks=5,
                            st=2,
                            in_c=256,
                            out_c=512,
                            activation="leaky_relu",
                            norm="instance"),
            conv_downsample(ks=5,
                            st=2,
                            in_c=512,
                            out_c=512,
                            activation="leaky_relu",
                            norm="instance"),
            conv_downsample(ks=5,
                            st=2,
                            in_c=512,
                            out_c=512,
                            activation="leaky_relu",
                            norm="instance"),
            conv_downsample(ks=5,
                            st=2,
                            in_c=512,
                            out_c=512,
                            activation="leaky_relu",
                            norm="instance"),
            conv_downsample(ks=3, st=2, in_c=512, out_c=512, activation="leaky_relu", norm=None),
        ])
        self.up_stack = nn.ModuleList([
            conv_upsample(ks=3,
                          st=2,
                          in_c=512,
                          out_c=512,
                          activation="leaky_relu",
                          dropout=True,
                          norm="instance"),
            conv_upsample(ks=5,
                          st=2,
                          in_c=1024,
                          out_c=512,
                          activation="leaky_relu",
                          dropout=True,
                          norm="instance"),
            conv_upsample(ks=5,
                          st=2,
                          in_c=1024,
                          out_c=512,
                          activation="leaky_relu",
                          dropout=True,
                          norm="instance"),
            conv_upsample(ks=5,
                          st=2,
                          in_c=1024,
                          out_c=512,
                          activation="leaky_relu",
                          norm="instance"),
            conv_upsample(ks=5,
                          st=2,
                          in_c=1024,
                          out_c=256,
                          activation="leaky_relu",
                          norm="instance"),
            conv_upsample(ks=5, st=2, in_c=512, out_c=128, activation="leaky_relu",
                          norm="instance"),
            conv_upsample(ks=5, st=2, in_c=256, out_c=64, activation="leaky_relu", norm="instance"),
        ])
        self.last = conv_upsample(ks=5, st=2, in_c=128, out_c=3, activation="tanh", norm=None)

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

        self.first_conv = conv_downsample(ks=3,
                                          st=1,
                                          in_c=input_channels,
                                          out_c=64,
                                          activation="leaky_relu",
                                          norm="batch")
        self.encoder = self.get_encoder()
        self.first_upsample = conv_upsample(ks=3,
                                            st=2,
                                            in_c=256,
                                            out_c=256,
                                            activation="leaky_relu",
                                            norm="batch")
        self.decoder = self.get_decoder()
        #self.final_conv = conv_downsample(ks=3,
        #                                  st=1,
        #                                  in_c=32,
        #                                  out_c=3,
        #                                  activation="tanh",
        #                                  norm=None)
        """
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
                                       activation="tanh",
                                       norm=None)
        """

    def _get_res_downsample(self, l, k, in_channels, out_channels, kernel_size, activation, norm):
        return torch.nn.Sequential(
            ResidualLayer(l=l,
                          k=k,
                          in_channels=in_channels,
                          kernel_size=kernel_size,
                          activation=activation,
                          norm=norm),
            conv_downsample(ks=kernel_size,
                            st=2,
                            in_c=in_channels,
                            out_c=out_channels,
                            activation=activation,
                            norm=norm))

    def _get_res_upsample(self, l, k, in_channels, out_channels, kernel_size, activation, norm):
        return torch.nn.Sequential(
            ResidualLayer(l=l,
                          k=k,
                          in_channels=in_channels,
                          kernel_size=kernel_size,
                          activation=activation,
                          norm=norm),
            conv_upsample(ks=kernel_size,
                          st=2,
                          in_c=in_channels,
                          out_c=out_channels,
                          activation=activation,
                          norm=norm))

    def get_encoder(self):
        return torch.nn.ModuleList([
            # 64,256,256
            self._get_res_downsample(l=2,
                                     k=1,
                                     in_channels=64,
                                     out_channels=128,
                                     kernel_size=3,
                                     activation="leaky_relu",
                                     norm="batch"),
            # 128,128,128
            self._get_res_downsample(l=2,
                                     k=1,
                                     in_channels=128,
                                     out_channels=256,
                                     kernel_size=3,
                                     activation="leaky_relu",
                                     norm="batch"),
            # 256,64,64
            self._get_res_downsample(l=2,
                                     k=1,
                                     in_channels=256,
                                     out_channels=256,
                                     kernel_size=3,
                                     activation="leaky_relu",
                                     norm="batch"),
            # 256,32,32
            self._get_res_downsample(l=2,
                                     k=1,
                                     in_channels=256,
                                     out_channels=256,
                                     kernel_size=3,
                                     activation="leaky_relu",
                                     norm="batch"),
            # 256,16,16
            self._get_res_downsample(l=2,
                                     k=1,
                                     in_channels=256,
                                     out_channels=256,
                                     kernel_size=3,
                                     activation="leaky_relu",
                                     norm="batch"),
            # 256,8,8
            self._get_res_downsample(l=2,
                                     k=1,
                                     in_channels=256,
                                     out_channels=256,
                                     kernel_size=3,
                                     activation="leaky_relu",
                                     norm="batch"),
            # 256,4,4
            self._get_res_downsample(l=2,
                                     k=1,
                                     in_channels=256,
                                     out_channels=256,
                                     kernel_size=3,
                                     activation="leaky_relu",
                                     norm="batch")
        ])
        """
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
            downsample = conv_downsample(ks=3, st=2, in_c=in_ch, out_c=f, activation="leaky_relu")
            layers.append(torch.nn.Sequential(layer, downsample))
            in_ch = f
            self.downsample_filters.append(f)
        self.downsample_filters = list(reversed(self.downsample_filters))
        return torch.nn.ModuleList(layers)
        """

    def get_decoder(self):
        """
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
        """
        return torch.nn.ModuleList([
            # 256,4,4 + 256,4,4
            self._get_res_upsample(l=2,
                                   k=1,
                                   in_channels=512,
                                   out_channels=256,
                                   kernel_size=3,
                                   activation="leaky_relu",
                                   norm="batch"),
            # 256,8,8 + 256,8,8
            self._get_res_upsample(l=2,
                                   k=1,
                                   in_channels=512,
                                   out_channels=256,
                                   kernel_size=3,
                                   activation="leaky_relu",
                                   norm="batch"),
            # 256,16,16 + 256,16,16
            self._get_res_upsample(l=2,
                                   k=1,
                                   in_channels=512,
                                   out_channels=256,
                                   kernel_size=3,
                                   activation="leaky_relu",
                                   norm="batch"),
            # 256,32,32 + 256,32,32
            self._get_res_upsample(l=2,
                                   k=1,
                                   in_channels=512,
                                   out_channels=256,
                                   kernel_size=3,
                                   activation="leaky_relu",
                                   norm="batch"),
            # 256,64,64 + 256,64,64
            self._get_res_upsample(l=2,
                                   k=1,
                                   in_channels=512,
                                   out_channels=128,
                                   kernel_size=3,
                                   activation="leaky_relu",
                                   norm="batch"),
            # 128,128,128 + 128,128,128
            conv_upsample(ks=3, st=2, in_c=256, out_c=3, activation="tanh", norm=None),
            # 3,256,256
        ])

    def forward(self, x):
        x = self.first_conv(x)
        d_feats = []
        for l in self.encoder:
            x = l(x)
            d_feats.append(x)
        d_feats.reverse()
        d_feats.pop(0)
        x = self.first_upsample(x)
        for i, l in enumerate(self.decoder):
            print(x.shape, d_feats[i].shape)
            x = torch.cat((x, d_feats[i]), 1)
            x = l(x)
        return x