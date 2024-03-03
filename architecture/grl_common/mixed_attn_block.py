import math
from abc import ABC
from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.grl_common.ops import (
    bchw_to_bhwc,
    bchw_to_blc,
    blc_to_bchw,
    blc_to_bhwc,
    calculate_mask,
    calculate_mask_all,
    get_relative_coords_table_all,
    get_relative_position_index_simple,
    window_partition,
    window_reverse,
)
from architecture.grl_common.swin_v1_block import Mlp
from timm.models.layers import DropPath


class CPB_MLP(nn.Sequential):
    def __init__(self, in_channels, out_channels, channels=512):
        m = [
            nn.Linear(in_channels, channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channels, out_channels, bias=False),
        ]
        super(CPB_MLP, self).__init__(*m)


class AffineTransformWindow(nn.Module):
    r"""Affine transformation of the attention map.
    The window is a square window.
    Supports attention between different window sizes
    """

    def __init__(
        self,
        num_heads,
        input_resolution,
        window_size,
        pretrained_window_size=[0, 0],
        shift_size=0,
        anchor_window_down_factor=1,
        args=None,
    ):
        super(AffineTransformWindow, self).__init__()
        # print("AffineTransformWindow", args)
        self.num_heads = num_heads
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.shift_size = shift_size
        self.anchor_window_down_factor = anchor_window_down_factor
        self.use_buffer = args.use_buffer

        logit_scale = torch.log(10 * torch.ones((num_heads, 1, 1)))
        self.logit_scale = nn.Parameter(logit_scale, requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = CPB_MLP(2, num_heads)
        if self.use_buffer:
            table = get_relative_coords_table_all(
                window_size, pretrained_window_size, anchor_window_down_factor
            )
            index = get_relative_position_index_simple(
                window_size, anchor_window_down_factor
            )
            self.register_buffer("relative_coords_table", table)
            self.register_buffer("relative_position_index", index)

            if self.shift_size > 0:
                attn_mask = calculate_mask(
                    input_resolution, self.window_size, self.shift_size
                )
            else:
                attn_mask = None
            self.register_buffer("attn_mask", attn_mask)

    def forward(self, attn, x_size):
        B_, H, N, _ = attn.shape
        device = attn.device
        # logit scale
        attn = attn * torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()

        # relative position bias
        if self.use_buffer:
            table = self.relative_coords_table
            index = self.relative_position_index
        else:
            table = get_relative_coords_table_all(
                self.window_size,
                self.pretrained_window_size,
                self.anchor_window_down_factor,
            ).to(device)
            index = get_relative_position_index_simple(
                self.window_size, self.anchor_window_down_factor
            ).to(device)

        bias_table = self.cpb_mlp(table)  # 2*Wh-1, 2*Ww-1, num_heads
        bias_table = bias_table.view(-1, self.num_heads)

        win_dim = prod(self.window_size)
        bias = bias_table[index.view(-1)]
        bias = bias.view(win_dim, win_dim, -1).permute(2, 0, 1).contiguous()
        # nH, Wh*Ww, Wh*Ww
        bias = 16 * torch.sigmoid(bias)
        attn = attn + bias.unsqueeze(0)

        # W-MSA/SW-MSA
        if self.use_buffer:
            mask = self.attn_mask
            # during test and window shift, recalculate the mask
            if self.input_resolution != x_size and self.shift_size > 0:
                mask = calculate_mask(x_size, self.window_size, self.shift_size)
                mask = mask.to(attn.device)
        else:
            if self.shift_size > 0:
                mask = calculate_mask(x_size, self.window_size, self.shift_size)
                mask = mask.to(attn.device)
            else:
                mask = None

        # shift attention mask
        if mask is not None:
            nW = mask.shape[0]
            mask = mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask
            attn = attn.view(-1, self.num_heads, N, N)

        return attn


class AffineTransformStripe(nn.Module):
    r"""Affine transformation of the attention map.
    The window is a stripe window. Supports attention between different window sizes
    """

    def __init__(
        self,
        num_heads,
        input_resolution,
        stripe_size,
        stripe_groups,
        stripe_shift,
        pretrained_stripe_size=[0, 0],
        anchor_window_down_factor=1,
        window_to_anchor=True,
        args=None,
    ):
        super(AffineTransformStripe, self).__init__()
        self.num_heads = num_heads
        self.input_resolution = input_resolution
        self.stripe_size = stripe_size
        self.stripe_groups = stripe_groups
        self.pretrained_stripe_size = pretrained_stripe_size
        # TODO: be careful when determining the pretrained_stripe_size
        self.stripe_shift = stripe_shift
        stripe_size, shift_size = self._get_stripe_info(input_resolution)
        self.anchor_window_down_factor = anchor_window_down_factor
        self.window_to_anchor = window_to_anchor
        self.use_buffer = args.use_buffer

        logit_scale = torch.log(10 * torch.ones((num_heads, 1, 1)))
        self.logit_scale = nn.Parameter(logit_scale, requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = CPB_MLP(2, num_heads)
        if self.use_buffer:
            table = get_relative_coords_table_all(
                stripe_size, pretrained_stripe_size, anchor_window_down_factor
            )
            index = get_relative_position_index_simple(
                stripe_size, anchor_window_down_factor, window_to_anchor
            )
            self.register_buffer("relative_coords_table", table)
            self.register_buffer("relative_position_index", index)

            if self.stripe_shift:
                attn_mask = calculate_mask_all(
                    input_resolution,
                    stripe_size,
                    shift_size,
                    anchor_window_down_factor,
                    window_to_anchor,
                )
            else:
                attn_mask = None
            self.register_buffer("attn_mask", attn_mask)

    def forward(self, attn, x_size):
        B_, H, N1, N2 = attn.shape
        device = attn.device
        # logit scale
        attn = attn * torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()

        # relative position bias
        stripe_size, shift_size = self._get_stripe_info(x_size)
        fixed_stripe_size = (
            self.stripe_groups[0] is None and self.stripe_groups[1] is None
        )
        if not self.use_buffer or (
            self.use_buffer
            and self.input_resolution != x_size
            and not fixed_stripe_size
        ):
            # during test and stripe size is not fixed.
            pretrained_stripe_size = (
                self.pretrained_stripe_size
            )  # or stripe_size; Needs further pondering
            table = get_relative_coords_table_all(
                stripe_size, pretrained_stripe_size, self.anchor_window_down_factor
            )
            table = table.to(device)
            index = get_relative_position_index_simple(
                stripe_size, self.anchor_window_down_factor, self.window_to_anchor
            ).to(device)
        else:
            table = self.relative_coords_table
            index = self.relative_position_index
        # The same table size-> 1, Wh+AWh-1, Ww+AWw-1, 2
        # But different index size -> # Wh*Ww, AWh*AWw
        # if N1 < N2:
        #     index = index.transpose(0, 1)

        bias_table = self.cpb_mlp(table).view(-1, self.num_heads)
        # if not self.training:
        #     print(bias_table.shape, index.max(), index.min())
        bias = bias_table[index.view(-1)]
        bias = bias.view(N1, N2, -1).permute(2, 0, 1).contiguous()
        # nH, Wh*Ww, Wh*Ww
        bias = 16 * torch.sigmoid(bias)
        # print(N1, N2, attn.shape, bias.unsqueeze(0).shape)
        attn = attn + bias.unsqueeze(0)

        # W-MSA/SW-MSA
        if self.use_buffer:
            mask = self.attn_mask
            # during test and window shift, recalculate the mask
            if self.input_resolution != x_size and self.stripe_shift > 0:
                mask = calculate_mask_all(
                    x_size,
                    stripe_size,
                    shift_size,
                    self.anchor_window_down_factor,
                    self.window_to_anchor,
                )
                mask = mask.to(device)
        else:
            if self.stripe_shift > 0:
                mask = calculate_mask_all(
                    x_size,
                    stripe_size,
                    shift_size,
                    self.anchor_window_down_factor,
                    self.window_to_anchor,
                )
                mask = mask.to(attn.device)
            else:
                mask = None

        # shift attention mask
        if mask is not None:
            nW = mask.shape[0]
            mask = mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B_ // nW, nW, self.num_heads, N1, N2) + mask
            attn = attn.view(-1, self.num_heads, N1, N2)

        return attn

    def _get_stripe_info(self, input_resolution):
        stripe_size, shift_size = [], []
        for s, g, d in zip(self.stripe_size, self.stripe_groups, input_resolution):
            if g is None:
                stripe_size.append(s)
                shift_size.append(s // 2 if self.stripe_shift else 0)
            else:
                stripe_size.append(d // g)
                shift_size.append(0 if g == 1 else d // (g * 2))
        return stripe_size, shift_size


class Attention(ABC, nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def attn(self, q, k, v, attn_transform, x_size, reshape=True):
        # cosine attention map
        B_, _, H, head_dim = q.shape
        if self.euclidean_dist:
            attn = torch.norm(q.unsqueeze(-2) - k.unsqueeze(-3), dim=-1)
        else:
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        attn = attn_transform(attn, x_size)
        # attention
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v  # B_, H, N1, head_dim
        if reshape:
            x = x.transpose(1, 2).reshape(B_, -1, H * head_dim)
        # B_, N, C
        return x


class WindowAttention(Attention):
    r"""Window attention. QKV is the input to the forward method.
    Args:
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
        self,
        input_resolution,
        window_size,
        num_heads,
        window_shift=False,
        attn_drop=0.0,
        pretrained_window_size=[0, 0],
        args=None,
    ):

        super(WindowAttention, self).__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.shift_size = window_size[0] // 2 if window_shift else 0
        self.euclidean_dist = args.euclidean_dist

        self.attn_transform = AffineTransformWindow(
            num_heads,
            input_resolution,
            window_size,
            pretrained_window_size,
            self.shift_size,
            args=args,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, x_size):
        """
        Args:
            qkv: input QKV features with shape of (B, L, 3C)
            x_size: use x_size to determine whether the relative positional bias table and index
            need to be regenerated.
        """
        H, W = x_size
        B, L, C = qkv.shape
        qkv = qkv.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            qkv = torch.roll(
                qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )

        # partition windows
        qkv = window_partition(qkv, self.window_size)  # nW*B, wh, ww, C
        qkv = qkv.view(-1, prod(self.window_size), C)  # nW*B, wh*ww, C

        B_, N, _ = qkv.shape
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # attention
        x = self.attn(q, k, v, self.attn_transform, x_size)

        # merge windows
        x = x.view(-1, *self.window_size, C // 3)
        x = window_reverse(x, self.window_size, x_size)  # B, H, W, C/3

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(B, L, C // 3)

        return x

    def extra_repr(self) -> str:
        return (
            f"window_size={self.window_size}, shift_size={self.shift_size}, "
            f"pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}"
        )

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class StripeAttention(Attention):
    r"""Stripe attention
    Args:
        stripe_size (tuple[int]): The height and width of the stripe.
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        pretrained_stripe_size (tuple[int]): The height and width of the stripe in pre-training.
    """

    def __init__(
        self,
        input_resolution,
        stripe_size,
        stripe_groups,
        stripe_shift,
        num_heads,
        attn_drop=0.0,
        pretrained_stripe_size=[0, 0],
        args=None,
    ):

        super(StripeAttention, self).__init__()
        self.input_resolution = input_resolution
        self.stripe_size = stripe_size  # Wh, Ww
        self.stripe_groups = stripe_groups
        self.stripe_shift = stripe_shift
        self.num_heads = num_heads
        self.pretrained_stripe_size = pretrained_stripe_size
        self.euclidean_dist = args.euclidean_dist

        self.attn_transform = AffineTransformStripe(
            num_heads,
            input_resolution,
            stripe_size,
            stripe_groups,
            stripe_shift,
            pretrained_stripe_size,
            anchor_window_down_factor=1,
            args=args,
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, x_size):
        """
        Args:
            x: input features with shape of (B, L, C)
            stripe_size: use stripe_size to determine whether the relative positional bias table and index
            need to be regenerated.
        """
        H, W = x_size
        B, L, C = qkv.shape
        qkv = qkv.view(B, H, W, C)

        running_stripe_size, running_shift_size = self.attn_transform._get_stripe_info(
            x_size
        )
        # cyclic shift
        if self.stripe_shift:
            qkv = torch.roll(
                qkv,
                shifts=(-running_shift_size[0], -running_shift_size[1]),
                dims=(1, 2),
            )

        # partition windows
        qkv = window_partition(qkv, running_stripe_size)  # nW*B, wh, ww, C
        qkv = qkv.view(-1, prod(running_stripe_size), C)  # nW*B, wh*ww, C

        B_, N, _ = qkv.shape
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # attention
        x = self.attn(q, k, v, self.attn_transform, x_size)

        # merge windows
        x = x.view(-1, *running_stripe_size, C // 3)
        x = window_reverse(x, running_stripe_size, x_size)  # B H W C/3

        # reverse the shift
        if self.stripe_shift:
            x = torch.roll(x, shifts=running_shift_size, dims=(1, 2))

        x = x.view(B, L, C // 3)
        return x

    def extra_repr(self) -> str:
        return (
            f"stripe_size={self.stripe_size}, stripe_groups={self.stripe_groups}, stripe_shift={self.stripe_shift}, "
            f"pretrained_stripe_size={self.pretrained_stripe_size}, num_heads={self.num_heads}"
        )

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class AnchorStripeAttention(Attention):
    r"""Stripe attention
    Args:
        stripe_size (tuple[int]): The height and width of the stripe.
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        pretrained_stripe_size (tuple[int]): The height and width of the stripe in pre-training.
    """

    def __init__(
        self,
        input_resolution,
        stripe_size,
        stripe_groups,
        stripe_shift,
        num_heads,
        attn_drop=0.0,
        pretrained_stripe_size=[0, 0],
        anchor_window_down_factor=1,
        args=None,
    ):

        super(AnchorStripeAttention, self).__init__()
        self.input_resolution = input_resolution
        self.stripe_size = stripe_size  # Wh, Ww
        self.stripe_groups = stripe_groups
        self.stripe_shift = stripe_shift
        self.num_heads = num_heads
        self.pretrained_stripe_size = pretrained_stripe_size
        self.anchor_window_down_factor = anchor_window_down_factor
        self.euclidean_dist = args.euclidean_dist

        self.attn_transform1 = AffineTransformStripe(
            num_heads,
            input_resolution,
            stripe_size,
            stripe_groups,
            stripe_shift,
            pretrained_stripe_size,
            anchor_window_down_factor,
            window_to_anchor=False,
            args=args,
        )

        self.attn_transform2 = AffineTransformStripe(
            num_heads,
            input_resolution,
            stripe_size,
            stripe_groups,
            stripe_shift,
            pretrained_stripe_size,
            anchor_window_down_factor,
            window_to_anchor=True,
            args=args,
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, anchor, x_size):
        """
        Args:
            qkv: input features with shape of (B, L, C)
            anchor:
            x_size: use stripe_size to determine whether the relative positional bias table and index
            need to be regenerated.
        """
        H, W = x_size
        B, L, C = qkv.shape
        qkv = qkv.view(B, H, W, C)

        stripe_size, shift_size = self.attn_transform1._get_stripe_info(x_size)
        anchor_stripe_size = [s // self.anchor_window_down_factor for s in stripe_size]
        anchor_shift_size = [s // self.anchor_window_down_factor for s in shift_size]
        # cyclic shift
        if self.stripe_shift:
            qkv = torch.roll(qkv, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            anchor = torch.roll(
                anchor,
                shifts=(-anchor_shift_size[0], -anchor_shift_size[1]),
                dims=(1, 2),
            )

        # partition windows
        qkv = window_partition(qkv, stripe_size)  # nW*B, wh, ww, C
        qkv = qkv.view(-1, prod(stripe_size), C)  # nW*B, wh*ww, C
        anchor = window_partition(anchor, anchor_stripe_size)
        anchor = anchor.view(-1, prod(anchor_stripe_size), C // 3)

        B_, N1, _ = qkv.shape
        N2 = anchor.shape[1]
        qkv = qkv.reshape(B_, N1, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        anchor = anchor.reshape(B_, N2, self.num_heads, -1).permute(0, 2, 1, 3)

        # attention
        x = self.attn(anchor, k, v, self.attn_transform1, x_size, False)
        x = self.attn(q, anchor, x, self.attn_transform2, x_size)

        # merge windows
        x = x.view(B_, *stripe_size, C // 3)
        x = window_reverse(x, stripe_size, x_size)  # B H' W' C

        # reverse the shift
        if self.stripe_shift:
            x = torch.roll(x, shifts=shift_size, dims=(1, 2))

        x = x.view(B, H * W, C // 3)
        return x

    def extra_repr(self) -> str:
        return (
            f"stripe_size={self.stripe_size}, stripe_groups={self.stripe_groups}, stripe_shift={self.stripe_shift}, "
            f"pretrained_stripe_size={self.pretrained_stripe_size}, num_heads={self.num_heads}, anchor_window_down_factor={self.anchor_window_down_factor}"
        )

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias, args):
        m = [
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                groups=in_channels,
                bias=bias,
            )
        ]
        if args.separable_conv_act:
            m.append(nn.GELU())
        m.append(nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias))
        super(SeparableConv, self).__init__(*m)


class QKVProjection(nn.Module):
    def __init__(self, dim, qkv_bias, proj_type, args):
        super(QKVProjection, self).__init__()
        self.proj_type = proj_type
        if proj_type == "linear":
            self.body = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.body = SeparableConv(dim, dim * 3, 3, 1, qkv_bias, args)

    def forward(self, x, x_size):
        if self.proj_type == "separable_conv":
            x = blc_to_bchw(x, x_size)
        x = self.body(x)
        if self.proj_type == "separable_conv":
            x = bchw_to_blc(x)
        return x


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)

        return x


class AnchorLinear(nn.Module):
    r"""Linear anchor projection layer
    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, in_channels, out_channels, down_factor, pooling_mode, bias):
        super().__init__()
        self.down_factor = down_factor
        if pooling_mode == "maxpool":
            self.pooling = nn.MaxPool2d(down_factor, down_factor)
        elif pooling_mode == "avgpool":
            self.pooling = nn.AvgPool2d(down_factor, down_factor)
        self.reduction = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        x = blc_to_bchw(x, x_size)
        x = bchw_to_blc(self.pooling(x))
        x = blc_to_bhwc(self.reduction(x), [s // self.down_factor for s in x_size])
        return x


class AnchorProjection(nn.Module):
    def __init__(self, dim, proj_type, one_stage, anchor_window_down_factor, args):
        super(AnchorProjection, self).__init__()
        self.proj_type = proj_type
        self.body = nn.ModuleList([])
        if one_stage:
            if proj_type == "patchmerging":
                m = PatchMerging(dim, dim // 2)
            elif proj_type == "conv2d":
                kernel_size = anchor_window_down_factor + 1
                stride = anchor_window_down_factor
                padding = kernel_size // 2
                m = nn.Conv2d(dim, dim // 2, kernel_size, stride, padding)
            elif proj_type == "separable_conv":
                kernel_size = anchor_window_down_factor + 1
                stride = anchor_window_down_factor
                m = SeparableConv(dim, dim // 2, kernel_size, stride, True, args)
            elif proj_type.find("pool") >= 0:
                m = AnchorLinear(
                    dim, dim // 2, anchor_window_down_factor, proj_type, True
                )
            self.body.append(m)
        else:
            for i in range(int(math.log2(anchor_window_down_factor))):
                cin = dim if i == 0 else dim // 2
                if proj_type == "patchmerging":
                    m = PatchMerging(cin, dim // 2)
                elif proj_type == "conv2d":
                    m = nn.Conv2d(cin, dim // 2, 3, 2, 1)
                elif proj_type == "separable_conv":
                    m = SeparableConv(cin, dim // 2, 3, 2, True, args)
                self.body.append(m)

    def forward(self, x, x_size):
        if self.proj_type.find("conv") >= 0:
            x = blc_to_bchw(x, x_size)
            for m in self.body:
                x = m(x)
            x = bchw_to_bhwc(x)
        elif self.proj_type.find("pool") >= 0:
            for m in self.body:
                x = m(x, x_size)
        else:
            for i, m in enumerate(self.body):
                x = m(x, [s // 2**i for s in x_size])
            x = blc_to_bhwc(x, [s // 2 ** (i + 1) for s in x_size])
        return x


class MixedAttention(nn.Module):
    r"""Mixed window attention and stripe attention
    Args:
        dim (int): Number of input channels.
        stripe_size (tuple[int]): The height and width of the stripe.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_stripe_size (tuple[int]): The height and width of the stripe in pre-training.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads_w,
        num_heads_s,
        window_size,
        window_shift,
        stripe_size,
        stripe_groups,
        stripe_shift,
        qkv_bias=True,
        qkv_proj_type="linear",
        anchor_proj_type="separable_conv",
        anchor_one_stage=True,
        anchor_window_down_factor=1,
        attn_drop=0.0,
        proj_drop=0.0,
        pretrained_window_size=[0, 0],
        pretrained_stripe_size=[0, 0],
        args=None,
    ):

        super(MixedAttention, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.use_anchor = anchor_window_down_factor > 1
        self.args = args
        # print(args)
        self.qkv = QKVProjection(dim, qkv_bias, qkv_proj_type, args)
        if self.use_anchor:
            # anchor is only used for stripe attention
            self.anchor = AnchorProjection(
                dim, anchor_proj_type, anchor_one_stage, anchor_window_down_factor, args
            )

        self.window_attn = WindowAttention(
            input_resolution,
            window_size,
            num_heads_w,
            window_shift,
            attn_drop,
            pretrained_window_size,
            args,
        )

        if self.args.double_window:
            self.stripe_attn = WindowAttention(
                input_resolution,
                window_size,
                num_heads_w,
                window_shift,
                attn_drop,
                pretrained_window_size,
                args,
            )
        else:
            if self.use_anchor:
                self.stripe_attn = AnchorStripeAttention(
                    input_resolution,
                    stripe_size,
                    stripe_groups,
                    stripe_shift,
                    num_heads_s,
                    attn_drop,
                    pretrained_stripe_size,
                    anchor_window_down_factor,
                    args,
                )
            else:
                if self.args.stripe_square:
                    self.stripe_attn = StripeAttention(
                        input_resolution,
                        window_size,
                        [None, None],
                        window_shift,
                        num_heads_s,
                        attn_drop,
                        pretrained_stripe_size,
                        args,
                    )
                else:
                    self.stripe_attn = StripeAttention(
                        input_resolution,
                        stripe_size,
                        stripe_groups,
                        stripe_shift,
                        num_heads_s,
                        attn_drop,
                        pretrained_stripe_size,
                        args,
                    )
        if self.args.out_proj_type == "linear":
            self.proj = nn.Linear(dim, dim)
        else:
            self.proj = nn.Conv2d(dim, dim, 3, 1, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_size):
        """
        Args:
            x: input features with shape of (B, L, C)
            stripe_size: use stripe_size to determine whether the relative positional bias table and index
            need to be regenerated.
        """
        B, L, C = x.shape

        # qkv projection
        qkv = self.qkv(x, x_size)
        qkv_window, qkv_stripe = torch.split(qkv, C * 3 // 2, dim=-1)
        # anchor projection
        if self.use_anchor:
            anchor = self.anchor(x, x_size)

        # attention
        x_window = self.window_attn(qkv_window, x_size)
        if self.use_anchor:
            x_stripe = self.stripe_attn(qkv_stripe, anchor, x_size)
        else:
            x_stripe = self.stripe_attn(qkv_stripe, x_size)
        x = torch.cat([x_window, x_stripe], dim=-1)

        # output projection
        if self.args.out_proj_type == "linear":
            x = self.proj(x)
        else:
            x = blc_to_bchw(x, x_size)
            x = bchw_to_blc(self.proj(x))
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        reduction (int): Channel reduction factor. Default: 16.
    """

    def __init__(self, num_feat, reduction=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // reduction, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // reduction, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=4, reduction=18):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, reduction),
        )

    def forward(self, x, x_size):
        x = self.cab(blc_to_bchw(x, x_size).contiguous())
        return bchw_to_blc(x)


class MixAttnTransformerBlock(nn.Module):
    r"""Mix attention transformer block with shared QKV projection and output projection for mixed attention modules.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_stripe_size (int): Window size in pre-training.
        attn_type (str, optional): Attention type. Default: cwhv.
                    c: residual blocks
                    w: window attention
                    h: horizontal stripe attention
                    v: vertical stripe attention
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads_w,
        num_heads_s,
        window_size=7,
        window_shift=False,
        stripe_size=[8, 8],
        stripe_groups=[None, None],
        stripe_shift=False,
        stripe_type="H",
        mlp_ratio=4.0,
        qkv_bias=True,
        qkv_proj_type="linear",
        anchor_proj_type="separable_conv",
        anchor_one_stage=True,
        anchor_window_down_factor=1,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=[0, 0],
        pretrained_stripe_size=[0, 0],
        res_scale=1.0,
        args=None,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads_w = num_heads_w
        self.num_heads_s = num_heads_s
        self.window_size = window_size
        self.window_shift = window_shift
        self.stripe_shift = stripe_shift
        self.stripe_type = stripe_type
        self.args = args
        if self.stripe_type == "W":
            self.stripe_size = stripe_size[::-1]
            self.stripe_groups = stripe_groups[::-1]
        else:
            self.stripe_size = stripe_size
            self.stripe_groups = stripe_groups
        self.mlp_ratio = mlp_ratio
        self.res_scale = res_scale

        self.attn = MixedAttention(
            dim,
            input_resolution,
            num_heads_w,
            num_heads_s,
            window_size,
            window_shift,
            self.stripe_size,
            self.stripe_groups,
            stripe_shift,
            qkv_bias,
            qkv_proj_type,
            anchor_proj_type,
            anchor_one_stage,
            anchor_window_down_factor,
            attn_drop,
            drop,
            pretrained_window_size,
            pretrained_stripe_size,
            args,
        )
        self.norm1 = norm_layer(dim)
        if self.args.local_connection:
            self.conv = CAB(dim)

    #         self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    #         self.mlp = Mlp(
    #             in_features=dim,
    #             hidden_features=int(dim * mlp_ratio),
    #             act_layer=act_layer,
    #             drop=drop,
    #         )
    #         self.norm2 = norm_layer(dim)

    def forward(self, x, x_size):
        # Mixed attention
        if self.args.local_connection:
            x = (
                x
                + self.res_scale * self.drop_path(self.norm1(self.attn(x, x_size)))
                + self.conv(x, x_size)
            )
        else:
            x = x + self.res_scale * self.drop_path(self.norm1(self.attn(x, x_size)))
        # FFN
        x = x + self.res_scale * self.drop_path(self.norm2(self.mlp(x)))

    #         return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads=({self.num_heads_w}, {self.num_heads_s}), "
            f"window_size={self.window_size}, window_shift={self.window_shift}, "
            f"stripe_size={self.stripe_size}, stripe_groups={self.stripe_groups}, stripe_shift={self.stripe_shift}, self.stripe_type={self.stripe_type}, "
            f"mlp_ratio={self.mlp_ratio}, res_scale={self.res_scale}"
        )


#     def flops(self):
#         flops = 0
#         H, W = self.input_resolution
#         # norm1
#         flops += self.dim * H * W
#         # W-MSA/SW-MSA
#         nW = H * W / self.stripe_size[0] / self.stripe_size[1]
#         flops += nW * self.attn.flops(self.stripe_size[0] * self.stripe_size[1])
#         # mlp
#         flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
#         # norm2
#         flops += self.dim * H * W
#         return flops
