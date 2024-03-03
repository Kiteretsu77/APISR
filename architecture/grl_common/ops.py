from math import prod
from typing import Tuple

import numpy as np
import torch
from timm.models.layers import to_2tuple


def bchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, C, H, W) to (B, H, W, C)."""
    return x.permute(0, 2, 3, 1)


def bhwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, H, W, C) to (B, C, H, W)."""
    return x.permute(0, 3, 1, 2)


def bchw_to_blc(x: torch.Tensor) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, C, H, W) to (B, L, C)."""
    return x.flatten(2).transpose(1, 2)


def blc_to_bchw(x: torch.Tensor, x_size: Tuple) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, C, H, W)."""
    B, L, C = x.shape
    return x.transpose(1, 2).view(B, C, *x_size)


def blc_to_bhwc(x: torch.Tensor, x_size: Tuple) -> torch.Tensor:
    """Rearrange a tensor from the shape (B, L, C) to (B, H, W, C)."""
    B, L, C = x.shape
    return x.view(B, *x_size, C)


def window_partition(x, window_size: Tuple[int, int]):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(
        B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size[0], window_size[1], C)
    )
    return windows


def window_reverse(windows, window_size: Tuple[int, int], img_size: Tuple[int, int]):
    """
    Args:
        windows: (num_windows * B, window_size[0], window_size[1], C)
        window_size (Tuple[int, int]): Window size
        img_size (Tuple[int, int]): Image size

    Returns:
        x: (B, H, W, C)
    """
    H, W = img_size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(
        B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def _fill_window(input_resolution, window_size, shift_size=None):
    if shift_size is None:
        shift_size = [s // 2 for s in window_size]

    img_mask = torch.zeros((1, *input_resolution, 1))  # 1 H W 1
    h_slices = (
        slice(0, -window_size[0]),
        slice(-window_size[0], -shift_size[0]),
        slice(-shift_size[0], None),
    )
    w_slices = (
        slice(0, -window_size[1]),
        slice(-window_size[1], -shift_size[1]),
        slice(-shift_size[1], None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, prod(window_size))
    return mask_windows


#####################################
# Different versions of the functions
# 1) Swin Transformer, SwinIR, Square window attention in GRL;
# 2) Early development of the decomposition-based efficient attention mechanism (efficient_win_attn.py);
# 3) GRL. Window-anchor attention mechanism.
# 1) & 3) are still useful
#####################################


def calculate_mask(input_resolution, window_size, shift_size):
    """
    Use case: 1)
    """
    # calculate attention mask for SW-MSA
    if isinstance(shift_size, int):
        shift_size = to_2tuple(shift_size)
    mask_windows = _fill_window(input_resolution, window_size, shift_size)

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )  # nW, window_size**2, window_size**2

    return attn_mask


def calculate_mask_all(
    input_resolution,
    window_size,
    shift_size,
    anchor_window_down_factor=1,
    window_to_anchor=True,
):
    """
    Use case: 3)
    """
    # calculate attention mask for SW-MSA
    anchor_resolution = [s // anchor_window_down_factor for s in input_resolution]
    aws = [s // anchor_window_down_factor for s in window_size]
    anchor_shift = [s // anchor_window_down_factor for s in shift_size]

    # mask of window1: nW, Wh**Ww
    mask_windows = _fill_window(input_resolution, window_size, shift_size)
    # mask of window2: nW, AWh*AWw
    mask_anchor = _fill_window(anchor_resolution, aws, anchor_shift)

    if window_to_anchor:
        attn_mask = mask_windows.unsqueeze(2) - mask_anchor.unsqueeze(1)
    else:
        attn_mask = mask_anchor.unsqueeze(2) - mask_windows.unsqueeze(1)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )  # nW, Wh**Ww, AWh*AWw

    return attn_mask


def calculate_win_mask(
    input_resolution1, input_resolution2, window_size1, window_size2
):
    """
    Use case: 2)
    """
    # calculate attention mask for SW-MSA

    # mask of window1: nW, Wh**Ww
    mask_windows1 = _fill_window(input_resolution1, window_size1)
    # mask of window2: nW, AWh*AWw
    mask_windows2 = _fill_window(input_resolution2, window_size2)

    attn_mask = mask_windows1.unsqueeze(2) - mask_windows2.unsqueeze(1)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )  # nW, Wh**Ww, AWh*AWw

    return attn_mask


def _get_meshgrid_coords(start_coords, end_coords):
    coord_h = torch.arange(start_coords[0], end_coords[0])
    coord_w = torch.arange(start_coords[1], end_coords[1])
    coords = torch.stack(torch.meshgrid([coord_h, coord_w], indexing="ij"))  # 2, Wh, Ww
    coords = torch.flatten(coords, 1)  # 2, Wh*Ww
    return coords


def get_relative_coords_table(
    window_size, pretrained_window_size=[0, 0], anchor_window_down_factor=1
):
    """
    Use case: 1)
    """
    # get relative_coords_table
    ws = window_size
    aws = [w // anchor_window_down_factor for w in window_size]
    pws = pretrained_window_size
    paws = [w // anchor_window_down_factor for w in pretrained_window_size]

    ts = [(w1 + w2) // 2 for w1, w2 in zip(ws, aws)]
    pts = [(w1 + w2) // 2 for w1, w2 in zip(pws, paws)]

    # TODO: pretrained window size and pretrained anchor window size is only used here.
    # TODO: Investigate whether it is really important to use this setting when finetuning large window size
    # TODO: based on pretrained weights with small window size.

    coord_h = torch.arange(-(ts[0] - 1), ts[0], dtype=torch.float32)
    coord_w = torch.arange(-(ts[1] - 1), ts[1], dtype=torch.float32)
    table = torch.stack(torch.meshgrid([coord_h, coord_w], indexing="ij")).permute(
        1, 2, 0
    )
    table = table.contiguous().unsqueeze(0)  # 1, Wh+AWh-1, Ww+AWw-1, 2
    if pts[0] > 0:
        table[:, :, :, 0] /= pts[0] - 1
        table[:, :, :, 1] /= pts[1] - 1
    else:
        table[:, :, :, 0] /= ts[0] - 1
        table[:, :, :, 1] /= ts[1] - 1
    table *= 8  # normalize to -8, 8
    table = torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / np.log2(8)
    return table


def get_relative_coords_table_all(
    window_size, pretrained_window_size=[0, 0], anchor_window_down_factor=1
):
    """
    Use case: 3)

    Support all window shapes.
    Args:
        window_size:
        pretrained_window_size:
        anchor_window_down_factor:

    Returns:

    """
    # get relative_coords_table
    ws = window_size
    aws = [w // anchor_window_down_factor for w in window_size]
    pws = pretrained_window_size
    paws = [w // anchor_window_down_factor for w in pretrained_window_size]

    # positive table size: (Ww - 1) - (Ww - AWw) // 2
    ts_p = [w1 - 1 - (w1 - w2) // 2 for w1, w2 in zip(ws, aws)]
    # negative table size: -(AWw - 1) - (Ww - AWw) // 2
    ts_n = [-(w2 - 1) - (w1 - w2) // 2 for w1, w2 in zip(ws, aws)]
    pts = [w1 - 1 - (w1 - w2) // 2 for w1, w2 in zip(pws, paws)]

    # TODO: pretrained window size and pretrained anchor window size is only used here.
    # TODO: Investigate whether it is really important to use this setting when finetuning large window size
    # TODO: based on pretrained weights with small window size.

    coord_h = torch.arange(ts_n[0], ts_p[0] + 1, dtype=torch.float32)
    coord_w = torch.arange(ts_n[1], ts_p[1] + 1, dtype=torch.float32)
    table = torch.stack(torch.meshgrid([coord_h, coord_w], indexing="ij")).permute(
        1, 2, 0
    )
    table = table.contiguous().unsqueeze(0)  # 1, Wh+AWh-1, Ww+AWw-1, 2
    if pts[0] > 0:
        table[:, :, :, 0] /= pts[0]
        table[:, :, :, 1] /= pts[1]
    else:
        table[:, :, :, 0] /= ts_p[0]
        table[:, :, :, 1] /= ts_p[1]
    table *= 8  # normalize to -8, 8
    table = torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / np.log2(8)
    # 1, Wh+AWh-1, Ww+AWw-1, 2
    return table


def coords_diff(coords1, coords2, max_diff):
    # The coordinates starts from (-start_coord[0], -start_coord[1])
    coords = coords1[:, :, None] - coords2[:, None, :]  # 2, Wh*Ww, AWh*AWw
    coords = coords.permute(1, 2, 0).contiguous()  # Wh*Ww, AWh*AWw, 2
    coords[:, :, 0] += max_diff[0] - 1  # shift to start from 0
    coords[:, :, 1] += max_diff[1] - 1
    coords[:, :, 0] *= 2 * max_diff[1] - 1
    idx = coords.sum(-1)  # Wh*Ww, AWh*AWw
    return idx


def get_relative_position_index(
    window_size, anchor_window_down_factor=1, window_to_anchor=True
):
    """
    Use case: 1)
    """
    # get pair-wise relative position index for each token inside the window
    ws = window_size
    aws = [w // anchor_window_down_factor for w in window_size]
    coords_anchor_end = [(w1 + w2) // 2 for w1, w2 in zip(ws, aws)]
    coords_anchor_start = [(w1 - w2) // 2 for w1, w2 in zip(ws, aws)]

    coords = _get_meshgrid_coords((0, 0), window_size)  # 2, Wh*Ww
    coords_anchor = _get_meshgrid_coords(coords_anchor_start, coords_anchor_end)
    # 2, AWh*AWw

    if window_to_anchor:
        idx = coords_diff(coords, coords_anchor, max_diff=coords_anchor_end)
    else:
        idx = coords_diff(coords_anchor, coords, max_diff=coords_anchor_end)
    return idx  # Wh*Ww, AWh*AWw or AWh*AWw, Wh*Ww


def coords_diff_odd(coords1, coords2, start_coord, max_diff):
    # The coordinates starts from (-start_coord[0], -start_coord[1])
    coords = coords1[:, :, None] - coords2[:, None, :]  # 2, Wh*Ww, AWh*AWw
    coords = coords.permute(1, 2, 0).contiguous()  # Wh*Ww, AWh*AWw, 2
    coords[:, :, 0] += start_coord[0]  # shift to start from 0
    coords[:, :, 1] += start_coord[1]
    coords[:, :, 0] *= max_diff
    idx = coords.sum(-1)  # Wh*Ww, AWh*AWw
    return idx


def get_relative_position_index_all(
    window_size, anchor_window_down_factor=1, window_to_anchor=True
):
    """
    Use case: 3)
    Support all window shapes:
        square window - square window
        rectangular window - rectangular window
        window - anchor
        anchor - window
        [8, 8] - [8, 8]
        [4, 86] - [2, 43]
    """
    # get pair-wise relative position index for each token inside the window
    ws = window_size
    aws = [w // anchor_window_down_factor for w in window_size]
    coords_anchor_start = [(w1 - w2) // 2 for w1, w2 in zip(ws, aws)]
    coords_anchor_end = [s + w2 for s, w2 in zip(coords_anchor_start, aws)]

    coords = _get_meshgrid_coords((0, 0), window_size)  # 2, Wh*Ww
    coords_anchor = _get_meshgrid_coords(coords_anchor_start, coords_anchor_end)
    # 2, AWh*AWw

    max_horizontal_diff = aws[1] + ws[1] - 1
    if window_to_anchor:
        offset = [w2 + s - 1 for s, w2 in zip(coords_anchor_start, aws)]
        idx = coords_diff_odd(coords, coords_anchor, offset, max_horizontal_diff)
    else:
        offset = [w1 - s - 1 for s, w1 in zip(coords_anchor_start, ws)]
        idx = coords_diff_odd(coords_anchor, coords, offset, max_horizontal_diff)
    return idx  # Wh*Ww, AWh*AWw or AWh*AWw, Wh*Ww


def get_relative_position_index_simple(
    window_size, anchor_window_down_factor=1, window_to_anchor=True
):
    """
    Use case: 3)
    This is a simplified version of get_relative_position_index_all
    The start coordinate of anchor window is also (0, 0)
    get pair-wise relative position index for each token inside the window
    """
    ws = window_size
    aws = [w // anchor_window_down_factor for w in window_size]

    coords = _get_meshgrid_coords((0, 0), window_size)  # 2, Wh*Ww
    coords_anchor = _get_meshgrid_coords((0, 0), aws)
    # 2, AWh*AWw

    max_horizontal_diff = aws[1] + ws[1] - 1
    if window_to_anchor:
        offset = [w2 - 1 for w2 in aws]
        idx = coords_diff_odd(coords, coords_anchor, offset, max_horizontal_diff)
    else:
        offset = [w1 - 1 for w1 in ws]
        idx = coords_diff_odd(coords_anchor, coords, offset, max_horizontal_diff)
    return idx  # Wh*Ww, AWh*AWw or AWh*AWw, Wh*Ww


# def get_relative_position_index(window_size):
#     # This is a very early version
#     # get pair-wise relative position index for each token inside the window
#     coords = _get_meshgrid_coords(start_coords=(0, 0), end_coords=window_size)

#     coords = coords[:, :, None] - coords[:, None, :]  # 2, Wh*Ww, Wh*Ww
#     coords = coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#     coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
#     coords[:, :, 1] += window_size[1] - 1
#     coords[:, :, 0] *= 2 * window_size[1] - 1
#     idx = coords.sum(-1)  # Wh*Ww, Wh*Ww
#     return idx


def get_relative_win_position_index(window_size, anchor_window_size):
    """
    Use case: 2)
    """
    # get pair-wise relative position index for each token inside the window
    ws = window_size
    aws = anchor_window_size
    coords_anchor_end = [(w1 + w2) // 2 for w1, w2 in zip(ws, aws)]
    coords_anchor_start = [(w1 - w2) // 2 for w1, w2 in zip(ws, aws)]

    coords = _get_meshgrid_coords((0, 0), window_size)  # 2, Wh*Ww
    coords_anchor = _get_meshgrid_coords(coords_anchor_start, coords_anchor_end)
    # 2, AWh*AWw
    coords = coords[:, :, None] - coords_anchor[:, None, :]  # 2, Wh*Ww, AWh*AWw
    coords = coords.permute(1, 2, 0).contiguous()  # Wh*Ww, AWh*AWw, 2
    coords[:, :, 0] += coords_anchor_end[0] - 1  # shift to start from 0
    coords[:, :, 1] += coords_anchor_end[1] - 1
    coords[:, :, 0] *= 2 * coords_anchor_end[1] - 1
    idx = coords.sum(-1)  # Wh*Ww, AWh*AWw
    return idx


# def get_relative_coords_table(window_size, pretrained_window_size):
#     # This is a very early version
#     # get relative_coords_table
#     ws = window_size
#     pws = pretrained_window_size
#     coord_h = torch.arange(-(ws[0] - 1), ws[0], dtype=torch.float32)
#     coord_w = torch.arange(-(ws[1] - 1), ws[1], dtype=torch.float32)
#     table = torch.stack(torch.meshgrid([coord_h, coord_w], indexing='ij')).permute(1, 2, 0)
#     table = table.contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
#     if pws[0] > 0:
#         table[:, :, :, 0] /= pws[0] - 1
#         table[:, :, :, 1] /= pws[1] - 1
#     else:
#         table[:, :, :, 0] /= ws[0] - 1
#         table[:, :, :, 1] /= ws[1] - 1
#     table *= 8  # normalize to -8, 8
#     table = torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / np.log2(8)
#     return table


def get_relative_win_coords_table(
    window_size,
    anchor_window_size,
    pretrained_window_size=[0, 0],
    pretrained_anchor_window_size=[0, 0],
):
    """
    Use case: 2)
    """
    # get relative_coords_table
    ws = window_size
    aws = anchor_window_size
    pws = pretrained_window_size
    paws = pretrained_anchor_window_size

    # TODO: pretrained window size and pretrained anchor window size is only used here.
    # TODO: Investigate whether it is really important to use this setting when finetuning large window size
    # TODO: based on pretrained weights with small window size.

    table_size = [(wsi + awsi) // 2 for wsi, awsi in zip(ws, aws)]
    table_size_pretrained = [(pwsi + pawsi) // 2 for pwsi, pawsi in zip(pws, paws)]
    coord_h = torch.arange(-(table_size[0] - 1), table_size[0], dtype=torch.float32)
    coord_w = torch.arange(-(table_size[1] - 1), table_size[1], dtype=torch.float32)
    table = torch.stack(torch.meshgrid([coord_h, coord_w], indexing="ij")).permute(
        1, 2, 0
    )
    table = table.contiguous().unsqueeze(0)  # 1, Wh+AWh-1, Ww+AWw-1, 2
    if table_size_pretrained[0] > 0:
        table[:, :, :, 0] /= table_size_pretrained[0] - 1
        table[:, :, :, 1] /= table_size_pretrained[1] - 1
    else:
        table[:, :, :, 0] /= table_size[0] - 1
        table[:, :, :, 1] /= table_size[1] - 1
    table *= 8  # normalize to -8, 8
    table = torch.sign(table) * torch.log2(torch.abs(table) + 1.0) / np.log2(8)
    return table


if __name__ == "__main__":
    table = get_relative_coords_table_all((4, 86), anchor_window_down_factor=2)
    table = table.view(-1, 2)
    index1 = get_relative_position_index_all((4, 86), 2, False)
    index2 = get_relative_position_index_simple((4, 86), 2, False)
    print(index2)
    index3 = get_relative_position_index_all((4, 86), 2)
    index4 = get_relative_position_index_simple((4, 86), 2)
    print(index4)
    print(
        table.shape,
        index2.shape,
        index2.max(),
        index2.min(),
        index4.shape,
        index4.max(),
        index4.min(),
        torch.allclose(index1, index2),
        torch.allclose(index3, index4),
    )

    table = get_relative_coords_table_all((4, 86), anchor_window_down_factor=1)
    table = table.view(-1, 2)
    index1 = get_relative_position_index_all((4, 86), 1, False)
    index2 = get_relative_position_index_simple((4, 86), 1, False)
    # print(index1)
    index3 = get_relative_position_index_all((4, 86), 1)
    index4 = get_relative_position_index_simple((4, 86), 1)
    # print(index2)
    print(
        table.shape,
        index2.shape,
        index2.max(),
        index2.min(),
        index4.shape,
        index4.max(),
        index4.min(),
        torch.allclose(index1, index2),
        torch.allclose(index3, index4),
    )

    table = get_relative_coords_table_all((8, 8), anchor_window_down_factor=2)
    table = table.view(-1, 2)
    index1 = get_relative_position_index_all((8, 8), 2, False)
    index2 = get_relative_position_index_simple((8, 8), 2, False)
    # print(index1)
    index3 = get_relative_position_index_all((8, 8), 2)
    index4 = get_relative_position_index_simple((8, 8), 2)
    # print(index2)
    print(
        table.shape,
        index2.shape,
        index2.max(),
        index2.min(),
        index4.shape,
        index4.max(),
        index4.min(),
        torch.allclose(index1, index2),
        torch.allclose(index3, index4),
    )

    table = get_relative_coords_table_all((8, 8), anchor_window_down_factor=1)
    table = table.view(-1, 2)
    index1 = get_relative_position_index_all((8, 8), 1, False)
    index2 = get_relative_position_index_simple((8, 8), 1, False)
    # print(index1)
    index3 = get_relative_position_index_all((8, 8), 1)
    index4 = get_relative_position_index_simple((8, 8), 1)
    # print(index2)
    print(
        table.shape,
        index2.shape,
        index2.max(),
        index2.min(),
        index4.shape,
        index4.max(),
        index4.min(),
        torch.allclose(index1, index2),
        torch.allclose(index3, index4),
    )
