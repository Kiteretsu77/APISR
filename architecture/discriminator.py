# -*- coding: utf-8 -*-
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import torch
import functools

class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):

        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out



def get_conv_layer(input_nc, ndf, kernel_size, stride, padding, bias=True, use_sn=False):
    if not use_sn:
        return nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    return spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))


class PatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator, the receptive field of default config is 70x70.

    Args:
        use_sn (bool): Use spectra_norm or not, if use_sn is True, then norm_type should be none.
    """

    def __init__(self,
                 num_in_ch,
                 num_feat=64,
                 num_layers=3,
                 max_nf_mult=8,
                 norm_type='batch',
                 use_sigmoid=False,
                 use_sn=False):
        super(PatchDiscriminator, self).__init__()

        norm_layer = self._get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            get_conv_layer(num_in_ch, num_feat, kernel_size=kw, stride=2, padding=padw, use_sn=use_sn),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, max_nf_mult)
            sequence += [
                get_conv_layer(
                    num_feat * nf_mult_prev,
                    num_feat * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                    use_sn=use_sn),
                norm_layer(num_feat * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**num_layers, max_nf_mult)
        sequence += [
            get_conv_layer(
                num_feat * nf_mult_prev,
                num_feat * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
                use_sn=use_sn),
            norm_layer(num_feat * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map 我觉得这个应该就是pixel by pixel的feedback反馈
        sequence += [get_conv_layer(num_feat * nf_mult, 1, kernel_size=kw, stride=1, padding=padw, use_sn=use_sn)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def _get_norm_layer(self, norm_type='batch'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm_type == 'batchnorm2d':
            norm_layer = nn.BatchNorm2d
        elif norm_type == 'none':
            norm_layer = nn.Identity
        else:
            raise NotImplementedError(f'normalization layer [{norm_type}] is not found')

        return norm_layer

    def forward(self, x):
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """Define a multi-scale discriminator, each discriminator is a instance of PatchDiscriminator.

    Args:
        num_layers (int or list): If the type of this variable is int, then degrade to PatchDiscriminator.
                                  If the type of this variable is list, then the length of the list is
                                  the number of discriminators.
        use_downscale (bool): Progressive downscale the input to feed into different discriminators.
                              If set to True, then the discriminators are usually the same.
    """

    def __init__(self,
                 num_in_ch,
                 num_feat=64,
                 num_layers=[3, 3, 3],
                 max_nf_mult=8,
                 norm_type='none',
                 use_sigmoid=False,
                 use_sn=True,
                 use_downscale=True):
        super(MultiScaleDiscriminator, self).__init__()

        if isinstance(num_layers, int):
            num_layers = [num_layers]

        # check whether the discriminators are the same
        if use_downscale:
            assert len(set(num_layers)) == 1
        self.use_downscale = use_downscale

        self.num_dis = len(num_layers)
        self.dis_list = nn.ModuleList()
        for nl in num_layers:
            self.dis_list.append(
                PatchDiscriminator(
                    num_in_ch,
                    num_feat=num_feat,
                    num_layers=nl,
                    max_nf_mult=max_nf_mult,
                    norm_type=norm_type,
                    use_sigmoid=use_sigmoid,
                    use_sn=use_sn,
                ))

    def forward(self, x):
        outs = []
        h, w = x.size()[2:]

        y = x
        for i in range(self.num_dis):
            if i != 0 and self.use_downscale:
                y = F.interpolate(y, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
                h, w = y.size()[2:]
            outs.append(self.dis_list[i](y))

        return outs


def main():
    from pthflops import count_ops
    from torchsummary import summary
    
    model = UNetDiscriminatorSN(3)
    pytorch_total_params = sum(p.numel() for p in model.parameters())

    # Create a network and a corresponding input
    device = 'cuda'
    inp = torch.rand(1, 3, 400, 400)

    # Count the number of FLOPs
    count_ops(model, inp)
    summary(model.cuda(), (3, 400, 400), batch_size=1)
    # print(f"pathGAN has param {pytorch_total_params//1000} K params")


if __name__ == "__main__":
    main()