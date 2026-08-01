"""Lightweight X-branch guidance gate at 1/16 resolution."""

import torch
import torch.nn as nn


class XGuide16(nn.Module):
    """Lightweight X-branch guidance module at 1/16.

    Input:  X-branch feature at 1/8 [B, C2, H/8, W/8]
    Output: (guidance feature, gate) used to modulate the RGB s4 feature at 1/16.

    Structure:
      - DWConv downsample (stride=2) + BN + ReLU
      - 1x1 Conv projecting to C3 channels + BN
      - 1x1 Conv + Sigmoid to produce the gate
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.down(x)     # [B, C3, H/16, W/16]
        gate = self.gate(feat)  # [B, C3, H/16, W/16]
        return feat, gate
