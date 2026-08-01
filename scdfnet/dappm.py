"""DAPPM3: 3-branch progressive aggregation context module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DAPPM3(nn.Module):
    """DAPPM3 context module (3-branch progressive aggregation).

    Branches:
      - Branch 0: local branch (1x1 Conv) -> b0
      - Branch 1: mid-scale pooled branch -> b1, with progressive aggregation
        b1 = progressive1(b1 + b0)
      - Branch 2: global pooled branch -> b2, with progressive aggregation
        b2 = progressive2(b2 + b1)
    The three branches are concatenated and compressed to out_channels.
    """

    def __init__(self, in_channels, out_channels=32, branch_channels=64, pool_mid=(2, 3)):
        super().__init__()

        # Branch 0: local
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 1: mid-scale pooling
        self.branch1_pool = nn.AdaptiveAvgPool2d(pool_mid)
        self.branch1_proj = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.branch1_refine = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        # Branch 2: global pooling
        self.branch2_pool = nn.AdaptiveAvgPool2d(1)
        self.branch2_proj = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2_refine = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        # Progressive aggregation layers
        self.progressive1 = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )
        self.progressive2 = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
        )

        # Compression: concat 3 branches -> out_channels
        self.compression = nn.Sequential(
            nn.Conv2d(branch_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Args: x [B, in_channels, H, W]. Returns: [B, out_channels, H, W]."""
        size = x.shape[-2:]

        b0 = self.branch0(x)

        b1 = self.branch1_pool(x)
        b1 = self.branch1_proj(b1)
        b1 = F.interpolate(b1, size=size, mode='bilinear', align_corners=True)
        b1 = self.branch1_refine(b1)
        b1 = self.progressive1(b1 + b0)

        b2 = self.branch2_pool(x)
        b2 = self.branch2_proj(b2)
        b2 = F.interpolate(b2, size=size, mode='bilinear', align_corners=True)
        b2 = self.branch2_refine(b2)
        b2 = self.progressive2(b2 + b1)

        out = torch.cat([b0, b1, b2], dim=1)
        out = self.compression(out)
        return out
