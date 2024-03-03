from math import prod

import torch
import torch.nn as nn
from architecture.grl_common.ops import (
    bchw_to_blc,
    blc_to_bchw,
    calculate_mask,
    window_partition,
    window_reverse,
)
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttentionV1(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        use_pe=True,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.use_pe = use_pe

        if self.use_pe:
            # define a parameter table of relative position bias
            ws = self.window_size
            table = torch.zeros((2 * ws[0] - 1) * (2 * ws[1] - 1), num_heads)
            self.relative_position_bias_table = nn.Parameter(table)
            # 2*Wh-1 * 2*Ww-1, nH
            trunc_normal_(self.relative_position_bias_table, std=0.02)

            self.get_relative_position_index(self.window_size)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def get_relative_position_index(self, window_size):
        # get pair-wise relative position index for each token inside the window
        coord_h = torch.arange(window_size[0])
        coord_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coord_h, coord_w]))  # 2, Wh, Ww
        coords = torch.flatten(coords, 1)  # 2, Wh*Ww
        coords = coords[:, :, None] - coords[:, None, :]  # 2, Wh*Ww, Wh*Ww
        coords = coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        coords[:, :, 1] += window_size[1] - 1
        coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape

        # qkv projection
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # attention map
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # positional encoding
        if self.use_pe:
            win_dim = prod(self.window_size)
            bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ]
            bias = bias.view(win_dim, win_dim, -1).permute(2, 0, 1).contiguous()
            # nH, Wh*Ww, Wh*Ww
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
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

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


class WindowAttentionWrapperV1(WindowAttentionV1):
    def __init__(self, shift_size, input_resolution, **kwargs):
        super(WindowAttentionWrapperV1, self).__init__(**kwargs)
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

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_mask = self.attn_mask
        else:
            attn_mask = calculate_mask(x_size, self.window_size, self.shift_size)
            attn_mask = attn_mask.to(x.device)

        # attention
        x = super(WindowAttentionWrapperV1, self).forward(x, mask=attn_mask)
        # nW*B, wh*ww, C

        # merge windows
        x = x.view(-1, *self.window_size, C)
        x = window_reverse(x, self.window_size, x_size)  # B, H, W, C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.view(B, H * W, C)

        return x


class SwinTransformerBlockV1(nn.Module):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
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
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
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

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttentionWrapperV1(
            shift_size=self.shift_size,
            input_resolution=self.input_resolution,
            dim=dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_pe=use_pe,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, x_size):
        # Window attention
        x = x + self.res_scale * self.drop_path(self.attn(self.norm1(x), x_size))
        # FFN
        x = x + self.res_scale * self.drop_path(self.mlp(self.norm2(x)))

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


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
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

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r"""Image to Patch Unembedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        B, C, H, W = x.shape
        x = bchw_to_blc(x)
        x = super(Linear, self).forward(x)
        x = blc_to_bchw(x, (H, W))
        return x


def build_last_conv(conv_type, dim):
    if conv_type == "1conv":
        block = nn.Conv2d(dim, dim, 3, 1, 1)
    elif conv_type == "3conv":
        # to save parameters and memory
        block = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(dim // 4, dim, 3, 1, 1),
        )
    elif conv_type == "1conv1x1":
        block = nn.Conv2d(dim, dim, 1, 1, 0)
    elif conv_type == "linear":
        block = Linear(dim, dim)
    return block


# class BasicLayer(nn.Module):
#     """A basic Swin Transformer layer for one stage.
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resolution.
#         depth (int): Number of blocks.
#         num_heads (int): Number of attention heads.
#         window_size (int): Local window size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         args: Additional arguments
#     """

#     def __init__(
#         self,
#         dim,
#         input_resolution,
#         depth,
#         num_heads,
#         window_size,
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         qk_scale=None,
#         drop=0.0,
#         attn_drop=0.0,
#         drop_path=0.0,
#         norm_layer=nn.LayerNorm,
#         downsample=None,
#         args=None,
#     ):

#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.depth = depth

#         # build blocks
#         self.blocks = nn.ModuleList(
#             [
#                 _parse_block(
#                     dim=dim,
#                     input_resolution=input_resolution,
#                     num_heads=num_heads,
#                     window_size=window_size,
#                     shift_size=0
#                     if args.no_shift
#                     else (0 if (i % 2 == 0) else window_size // 2),
#                     mlp_ratio=mlp_ratio,
#                     qkv_bias=qkv_bias,
#                     qk_scale=qk_scale,
#                     drop=drop,
#                     attn_drop=attn_drop,
#                     drop_path=drop_path[i]
#                     if isinstance(drop_path, list)
#                     else drop_path,
#                     norm_layer=norm_layer,
#                     stripe_type="H" if (i % 2 == 0) else "W",
#                     args=args,
#                 )
#                 for i in range(depth)
#             ]
#         )
#         # self.blocks = nn.ModuleList(
#         #     [
#         #         STV1Block(
#         #             dim=dim,
#         #             input_resolution=input_resolution,
#         #             num_heads=num_heads,
#         #             window_size=window_size,
#         #             shift_size=0 if (i % 2 == 0) else window_size // 2,
#         #             mlp_ratio=mlp_ratio,
#         #             qkv_bias=qkv_bias,
#         #             qk_scale=qk_scale,
#         #             drop=drop,
#         #             attn_drop=attn_drop,
#         #             drop_path=drop_path[i]
#         #             if isinstance(drop_path, list)
#         #             else drop_path,
#         #             norm_layer=norm_layer,
#         #         )
#         #         for i in range(depth)
#         #     ]
#         # )

#         # patch merging layer
#         if downsample is not None:
#             self.downsample = downsample(
#                 input_resolution, dim=dim, norm_layer=norm_layer
#             )
#         else:
#             self.downsample = None

#     def forward(self, x, x_size):
#         for blk in self.blocks:
#             x = blk(x, x_size)
#         if self.downsample is not None:
#             x = self.downsample(x)
#         return x

#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

#     def flops(self):
#         flops = 0
#         for blk in self.blocks:
#             flops += blk.flops()
#         if self.downsample is not None:
#             flops += self.downsample.flops()
#         return flops
