"""
EDSR common.py
Since a lot of models are developed on top of EDSR, here we include some common functions from EDSR.
In this repository, the common functions is used by edsr_esa.py and ipt.py
"""


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class MeanShift(nn.Conv2d):
    def __init__(
        self,
        rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040),
        rgb_std=(1.0, 1.0, 1.0),
        sign=-1,
    ):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
        self,
        conv,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=False,
        bn=True,
        act=nn.ReLU(True),
    ):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ESA(nn.Module):
    def __init__(self, esa_channels, n_feats):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        #         self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        #         self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        #         v_range = self.relu(self.conv_max(v_max))
        #         c3 = self.relu(self.conv3(v_range))
        #         c3 = self.conv3_(c3)
        c3 = F.interpolate(
            c3, (x.size(2), x.size(3)), mode="bilinear", align_corners=False
        )
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


# class ESA(nn.Module):
#     def __init__(self, esa_channels, n_feats, conv=nn.Conv2d):
#         super(ESA, self).__init__()
#         f = n_feats // 4
#         self.conv1 = conv(n_feats, f, kernel_size=1)
#         self.conv_f = conv(f, f, kernel_size=1)
#         self.conv_max = conv(f, f, kernel_size=3, padding=1)
#         self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
#         self.conv3 = conv(f, f, kernel_size=3, padding=1)
#         self.conv3_ = conv(f, f, kernel_size=3, padding=1)
#         self.conv4 = conv(f, n_feats, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         c1_ = (self.conv1(x))
#         c1 = self.conv2(c1_)
#         v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
#         v_range = self.relu(self.conv_max(v_max))
#         c3 = self.relu(self.conv3(v_range))
#         c3 = self.conv3_(c3)
#         c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
#         cf = self.conv_f(c1_)
#         c4 = self.conv4(c3 + cf)
#         m = self.sigmoid(c4)
#
#         return x * m


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
        esa_block=True,
        depth_wise_kernel=7,
    ):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.esa_block = esa_block
        if self.esa_block:
            esa_channels = 16
            self.c5 = nn.Conv2d(
                n_feats,
                n_feats,
                depth_wise_kernel,
                padding=depth_wise_kernel // 2,
                groups=n_feats,
                bias=True,
            )
            self.esa = ESA(esa_channels, n_feats)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        if self.esa_block:
            res = self.esa(self.c5(res))

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class LiteUpsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, n_out=3, bn=False, act=False, bias=True):

        m = []
        m.append(conv(n_feats, n_out * (scale**2), 3, bias))
        m.append(nn.PixelShuffle(scale))
        # if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
        #     for _ in range(int(math.log(scale, 2))):
        #         m.append(conv(n_feats, 4 * n_out, 3, bias))
        #         m.append(nn.PixelShuffle(2))
        #         if bn:
        #             m.append(nn.BatchNorm2d(n_out))
        #         if act == 'relu':
        #             m.append(nn.ReLU(True))
        #         elif act == 'prelu':
        #             m.append(nn.PReLU(n_out))

        # elif scale == 3:
        #     m.append(conv(n_feats, 9 * n_out, 3, bias))
        #     m.append(nn.PixelShuffle(3))
        #     if bn:
        #         m.append(nn.BatchNorm2d(n_out))
        #     if act == 'relu':
        #         m.append(nn.ReLU(True))
        #     elif act == 'prelu':
        #         m.append(nn.PReLU(n_out))
        # else:
        #     raise NotImplementedError

        super(LiteUpsampler, self).__init__(*m)
