"""CSDB: Cross-Scale Detail Bridge.

Injects the Fusion1 detail (x1, 1/4, 32ch) into the DGRF output (s3, 1/8, C2)
to supplement detail and boundary priors with a relation-aware gate and a
learnable scalar gamma.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBNAct


class CSDB(nn.Module):
    """Cross-Scale Detail Bridge (CSDB).

    Injects the Fusion1 detail (x1, 1/4, 32ch) into the DGRF output
    (s3, 1/8, C2) to supplement detail and boundary priors. This is a
    cross-scale bridge, not another cross-modal fusion.
    """

    def __init__(self, in_low=32, in_high=128, out_high=128, gamma_init=0.1):
        super().__init__()
        self.low_proj = ConvBNAct(in_low, out_high, kernel_size=1, padding=0, act_type='relu')
        self.gate = nn.Sequential(
            nn.Conv2d(out_high * 4, out_high // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_high // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_high // 2, out_high, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.gamma = nn.Parameter(torch.tensor(gamma_init))

    def forward(self, x1, s3):
        x1_down = F.interpolate(x1, size=s3.shape[-2:], mode='bilinear', align_corners=True)
        x1_proj = self.low_proj(x1_down)

        diff = torch.abs(x1_proj - s3)
        prod = x1_proj * s3
        feat = torch.cat([x1_proj, s3, diff, prod], dim=1)

        gate = self.gate(feat)
        out = s3 + self.gamma * gate * x1_proj
        return out
