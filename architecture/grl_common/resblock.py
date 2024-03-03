import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block without BN.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feats (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, num_feats=64, res_scale=1.0, bias=True, shortcut=True):
        super().__init__()
        self.res_scale = res_scale
        self.shortcut = shortcut
        self.conv1 = nn.Conv2d(num_feats, num_feats, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(num_feats, num_feats, 3, 1, 1, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        if self.shortcut:
            return identity + out * self.res_scale
        else:
            return out * self.res_scale


class ResBlockWrapper(ResBlock):
    "Used for transformers"

    def __init__(self, num_feats, bias=True, shortcut=True):
        super(ResBlockWrapper, self).__init__(
            num_feats=num_feats, bias=bias, shortcut=shortcut
        )

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = super(ResBlockWrapper, self).forward(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x
