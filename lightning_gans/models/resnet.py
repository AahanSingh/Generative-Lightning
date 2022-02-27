import torch

activations = {"relu": torch.nn.ReLU(inplace=True), "tanh": torch.nn.Tanh()}


def conv_downsample(ks, st, in_c, out_c, activation=None):
    layers = [
        torch.nn.Conv2d(in_c, out_c, ks, st, padding=ks // 2),
        torch.nn.BatchNorm2d(num_features=out_c)
    ]
    if activation:
        layers.append(activations[activation])
    return torch.nn.Sequential(*layers)


def conv_upsample(ks, st, in_c, out_c, activation=None):
    layers = [
        torch.nn.ConvTranspose2d(in_c, out_c, ks, st, padding=ks // 2, output_padding=1),
        torch.nn.BatchNorm2d(num_features=out_c),
    ]
    if activation:
        layers.append(activations[activation])
    return torch.nn.Sequential(*layers)


def wide_block(in_channels, res_channels, l, k, kernel_size):
    prev_out_channels = in_channels
    layers = []
    for _ in range(l - 1):
        layers.append(
            conv_downsample(ks=kernel_size, st=1, in_c=prev_out_channels, out_c=res_channels * k))
        prev_out_channels = res_channels * k
    layers.append(
        conv_downsample(ks=kernel_size,
                        st=1,
                        in_c=prev_out_channels,
                        out_c=in_channels,
                        activation=False))
    layers = torch.nn.Sequential(*layers)
    return layers


class ResidualLayer(torch.nn.Module):

    def __init__(self, l, k, in_channels, res_channels, kernel_size):
        super().__init__()
        self.wide_block = wide_block(in_channels=in_channels,
                                     res_channels=res_channels,
                                     l=l,
                                     k=k,
                                     kernel_size=kernel_size)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x_out = self.wide_block(x)
        x_out += x
        x_out = self.relu(x_out)
        return x_out


class WideResnetEncoder(torch.nn.Module):

    def __init__(self, l, k, input_channels, image_size):
        super().__init__()
        self.layers = []
        ks = 3
        res_channels = input_channels * 2
        while image_size > 2:
            layer = ResidualLayer(l=l,
                                  k=k,
                                  in_channels=input_channels,
                                  res_channels=res_channels,
                                  kernel_size=ks)
            dimension_reduction = conv_downsample(ks=ks,
                                                  st=2,
                                                  in_c=input_channels,
                                                  out_c=input_channels * 2,
                                                  activation="relu")
            self.layers.append(layer)
            self.layers.append(dimension_reduction)
            input_channels *= 2
            res_channels *= 2
            image_size //= 2

        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
        return x


class WideResnetDecoder(torch.nn.Module):

    def __init__(self, l, k, input_channels, image_size):
        super().__init__()
        self.layers = []
        ks = 3
        res_channels = input_channels // 2
        tmp = 2
        while tmp != image_size:
            layer = ResidualLayer(l=l,
                                  k=k,
                                  in_channels=input_channels,
                                  res_channels=res_channels,
                                  kernel_size=ks)
            self.layers.append(layer)
            upsampling = conv_upsample(ks=ks,
                                       st=2,
                                       in_c=input_channels,
                                       out_c=input_channels // 2,
                                       activation="relu")
            self.layers.append(upsampling)
            input_channels //= 2
            res_channels //= 2
            image_size //= 2
        self.layers.append(
            conv_downsample(ks=1, st=1, in_c=input_channels, out_c=3, activation="tanh"))
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x
