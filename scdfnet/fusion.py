"""Encoder fusion modules.

- encoder_fusion: cosine-attention cross-modal fusion used at Fusion1.
- DGRFGate / DGRF: Difference-Guided Residual Fusion (the second-stage fusion).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import fusion_attention


class encoder_fusion(nn.Module):
    """Standard cross-modal fusion module used at Fusion1."""

    def __init__(self, num_channel, hid_channels, pool_size, skip_conncetion, last_fusion):
        super(encoder_fusion, self).__init__()

        self.attention = fusion_attention(num_channel, hid_channels, pool_size)
        self.skip_conncetion = skip_conncetion
        self.last_fusion = last_fusion

    def forward(self, rgb, depth):
        wd, wr = self.attention(rgb, depth)
        w_rgb = rgb.mul(wr)
        w_depth = depth.mul(wd)

        if self.skip_conncetion == True and self.last_fusion == False:
            rgb = rgb + w_depth
            depth = depth + w_rgb
            SC = w_rgb + w_depth
            return rgb, depth, SC

        if self.skip_conncetion == False and self.last_fusion == False:
            rgb = rgb + w_depth
            depth = depth + w_rgb
            return rgb, depth

        # last_fusion=True
        SC = w_rgb + w_depth
        return SC


class DGRFGate(nn.Module):
    """Difference-aware residual gate (the gate inside DGRF).

    Replaces cosine similarity with a pooled-feature + difference + product
    based residual gate producing beta.
    """

    def __init__(self, num_channel, hid_channels, pool_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.gate = nn.Sequential(
            nn.Conv2d(num_channel * 4, hid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, num_channel, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, rgb, x):
        rgb_p = self.pool(rgb)
        x_p = self.pool(x)
        diff = torch.abs(rgb_p - x_p)
        prod = rgb_p * x_p
        feat = torch.cat([rgb_p, x_p, diff, prod], dim=1)
        beta = self.gate(feat)
        return beta


class DGRF(nn.Module):
    """Difference-Guided Residual Fusion (DGRF).

    beta = DGRFGate(rgb, x)
    SC  = rgb + Up(beta) * x
    """

    def __init__(self, num_channel, hid_channels, pool_size):
        super().__init__()
        self.attention = DGRFGate(num_channel, hid_channels, pool_size)

    def forward(self, rgb, x):
        beta = self.attention(rgb, x)
        beta = F.interpolate(beta, size=x.shape[-2:], mode='bilinear', align_corners=True)
        w_x = x * beta
        sc = rgb + w_x
        return sc
