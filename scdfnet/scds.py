"""SCDS: Saliency-Calibrated Discrepancy Suppression.

SFC-style dual-branch discrepancy suppression applied to the CMCA-aligned
features. Contains the SFCBranch / Fuse_sfc_dual calibrators. Part of CSPC.
"""

import math

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class SFCBranch(nn.Module):
    """SFC-style single-branch saliency calibrator.

    Extracts multi-scale saliency features and produces a calibration map.
    """

    def __init__(self, dim, r=16, L=32):
        super().__init__()
        d = max(dim // r, L)

        self.conv0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 5, stride=1, padding=4, groups=dim, dilation=2)

        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Conv2d(d, dim, 1, 1, bias=False)

        self.fuse = nn.Conv2d(dim // 2, dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)

        attn1_half = self.conv1(attn1)
        attn2_half = self.conv2(attn2)

        attn_cat = torch.cat([attn1_half, attn2_half], dim=1)

        ch_attn = self.global_pool(attn_cat)
        z = self.fc1(ch_attn)
        a_b = self.fc2(z)

        B, C, _, _ = a_b.shape
        a_b = a_b.view(B, 2, C // 2, 1)
        a_b = self.softmax(a_b)
        a1, a2 = a_b.chunk(2, dim=1)
        a1 = a1.view(B, C // 2, 1, 1)
        a2 = a2.view(B, C // 2, 1, 1)

        out = attn1_half * a1 + attn2_half * a2
        out = torch.sigmoid(self.fuse(out))
        return out


class Fuse_sfc_dual(nn.Module):
    """SFC-style dual-branch calibrator.

    Produces two calibration maps (calib1, calib2) from x1 and x2, combining
    each branch's saliency with their difference.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.sfc1 = SFCBranch(dim)
        self.sfc2 = SFCBranch(dim)

        self.cross_gate = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim * 2, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        s1 = self.sfc1(x1)
        s2 = self.sfc2(x2)
        diff = torch.abs(s1 - s2)

        gates = self.cross_gate(torch.cat([s1, s2, diff], dim=1))
        g1, g2 = gates.chunk(2, dim=1)

        calib1 = s1 * g1
        calib2 = s2 * g2
        return calib1, calib2


class SCDS(nn.Module):
    """Saliency-Calibrated Discrepancy Suppression (SCDS).

    SFC-style dual-branch discrepancy suppression applied to the CMCA-aligned
    features. Returns suppressed (out_1, out_2).
    """

    def __init__(self, dim, reduction=4):
        super(SCDS, self).__init__()
        self.fuse_sc = Fuse_sfc_dual(dim)
        self.sigmoid = nn.Sigmoid()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim * 2 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2 // reduction, dim),
            nn.Sigmoid())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B1, C1, H1, W1 = x1.shape
        x1_flat = x1.flatten(2).transpose(1, 2)
        x2_flat = x2.flatten(2).transpose(1, 2)
        mid_feature = self.gate(torch.cat((x1_flat, x2_flat), dim=2))
        mid_feature = mid_feature.reshape(B1, H1, W1, C1).permute(0, 3, 1, 2).contiguous()
        fusion1, fusion2 = self.fuse_sc(x1, x2)
        channel_feature = mid_feature * fusion1
        spatial_feature = mid_feature * fusion2
        out_x1 = x1 + channel_feature * x2
        out_x2 = x2 + spatial_feature * x1
        out_1 = self.sigmoid(out_x1 * channel_feature - out_x1) * out_x1 + out_x1 * channel_feature
        out_2 = self.sigmoid(out_x2 * spatial_feature - out_x2) * out_x2 + out_x2 * spatial_feature
        return out_1, out_2
