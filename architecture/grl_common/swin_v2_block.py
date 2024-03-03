import math
from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture.grl_common.ops import (
    calculate_mask,
    get_relative_coords_table,
    get_relative_position_index,
    window_partition,
    window_reverse,
)
from architecture.grl_common.swin_v1_block import Mlp
from timm.models.layers import DropPath, to_2tuple


class WindowAttentionV2(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        pretrained_window_size=[0, 0],
        use_pe=True,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.use_pe = use_pe

        self.logit_scale = nn.Parameter(
            torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
        )

        if self.use_pe:
            # mlp to generate continuous relative position bias
            self.cpb_mlp = nn.Sequential(
                nn.Linear(2, 512, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_heads, bias=False),
            )
            table = get_relative_coords_table(window_size, pretrained_window_size)
            index = get_relative_position_index(window_size)
            self.register_buffer("relative_coords_table", table)
            self.register_buffer("relative_position_index", index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.qkv = nn.Linear(dim, dim * 3, bias=False)
        # if qkv_bias:
        #     self.q_bias = nn.Parameter(torch.zeros(dim))
        #     self.v_bias = nn.Parameter(torch.zeros(dim))
        # else:
        #     self.q_bias = None
        #     self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        # qkv projection
        # qkv_bias = None
        # if self.q_bias is not None:
        #     qkv_bias = torch.cat(
        #         (
        #             self.q_bias,
        #             torch.zeros_like(self.v_bias, requires_grad=False),
        #             self.v_bias,
        #         )
        #     )
        # qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # cosine attention map
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1.0 / 0.01)).exp()
        attn = attn * logit_scale

        # positional encoding
        if self.use_pe:
            bias_table = self.cpb_mlp(self.relative_coords_table)
            bias_table = bias_table.view(-1, self.num_heads)

            win_dim = prod(self.window_size)
            bias = bias_table[self.relative_position_index.view(-1)]
            bias = bias.view(win_dim, win_dim, -1).permute(2, 0, 1).contiguous()
            # nH, Wh*Ww, Wh*Ww
            bias = 16 * torch.sigmoid(bias)
            attn = attn + bias.unsqueeze(0)

        # shift attention mask
        if mask is not None:
            nW = mask.shape[0]
            mask = mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask
            attn = attn.view(-1, self.num_heads, N, N)

        # attention
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, window_size={self.window_size}, "
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


class WindowAttentionWrapperV2(WindowAttentionV2):
    def __init__(self, shift_size, input_resolution, **kwargs):
        super(WindowAttentionWrapperV2, self).__init__(**kwargs)
        self.shift_size = shift_size
        self.input_resolution = input_resolution

        if self.shift_size > 0:
            attn_mask = calculate_mask(input_resolution, self.window_size, shift_size)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # partition windows
        x = window_partition(x, self.window_size)  # nW*B, wh, ww, C
        x = x.view(-1, prod(self.window_size), C)  # nW*B, wh*ww, C

        # W-MSA/SW-MSA
        if self.input_resolution == x_size:
            attn_mask = self.attn_mask
        else:
            attn_mask = calculate_mask(x_size, self.window_size, self.shift_size)
            attn_mask = attn_mask.to(x.device)

        # attention
        x = super(WindowAttentionWrapperV2, self).forward(x, mask=attn_mask)
        # nW*B, wh*ww, C

        # merge windows
        x = x.view(-1, *self.window_size, C)
        x = window_reverse(x, self.window_size, x_size)  # B, H, W, C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(B, H * W, C)

        return x


class SwinTransformerBlockV2(nn.Module):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        pretrained_window_size=0,
        use_pe=True,
        res_scale=1.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"
        self.res_scale = res_scale

        self.attn = WindowAttentionWrapperV2(
            shift_size=self.shift_size,
            input_resolution=self.input_resolution,
            dim=dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size),
            use_pe=use_pe,
        )
        self.norm1 = norm_layer(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.norm2 = norm_layer(dim)

    def forward(self, x, x_size):
        # Window attention
        x = x + self.res_scale * self.drop_path(self.norm1(self.attn(x, x_size)))
        # FFN
        x = x + self.res_scale * self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}, res_scale={self.res_scale}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops
