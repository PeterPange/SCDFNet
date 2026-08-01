"""SCDFNet segmentation decoder.

Contains the decoder fusion modules (cosine-attention based), the global
semantic gate, and the full Decoder that fuses context + multi-scale features
into the final segmentation logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBNAct, fusion_attention


class decoder_fusion(nn.Module):
    """Standard weighted decoder fusion."""

    def __init__(self, num_channel, hid_channels, pool_size):
        super(decoder_fusion, self).__init__()
        self.attention = fusion_attention(num_channel, hid_channels, pool_size)

    def forward(self, x_high, x_low):
        alpha, beta = self.attention(x_high, x_low)
        return x_high.mul(alpha) + x_low.mul(beta)


class decoder_fusion_residual_preserve(nn.Module):
    """Residual-preserve decoder fusion."""

    def __init__(self, num_channel, hid_channels, pool_size):
        super(decoder_fusion_residual_preserve, self).__init__()
        self.attention = fusion_attention(num_channel, hid_channels, pool_size)

    def forward(self, x_high, x_low):
        _, beta = self.attention(x_high, x_low)
        return x_high + x_low.mul(beta)


class SemanticGate(nn.Module):
    """Decoder global semantic gate with controllable strength."""

    def __init__(self, in_channels=32, x3_channels=64, x1_channels=32, hidden=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.gate3 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, x3_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.gate1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, x1_channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.alpha3 = nn.Parameter(torch.tensor(0.1))
        self.alpha1 = nn.Parameter(torch.tensor(0.1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x5_context):
        global_feat = self.pool(x5_context)
        g3 = self.gate3(global_feat)
        g1 = self.gate1(global_feat)
        return g3, g1


class Decoder(nn.Module):
    """SCDFNet segmentation decoder."""

    def __init__(self, decoder_channels=[32, 64, 128, 32, 14], act_type='relu',
                 pool_out=None, dataset="MFNet"):
        super().__init__()

        self.dataset = dataset

        if pool_out is None:
            if dataset == "Cityscapes":
                pool_out = [(16, 32), (8, 16), (4, 8), (2, 4)]
            elif dataset in ["MFNet", "ZJU", "FMB"]:
                pool_out = [(16, 24), (8, 12), (4, 6), (2, 3)]
            else:
                pool_out = [(8, 8), (4, 4), (2, 2), (1, 1)]

        self.CBR1 = ConvBNAct(decoder_channels[0], decoder_channels[0], act_type=act_type)
        self.CBR2 = ConvBNAct(decoder_channels[0], decoder_channels[1], act_type=act_type)
        self.CBR3 = ConvBNAct(decoder_channels[1], decoder_channels[2], act_type=act_type)
        self.CBR4 = ConvBNAct(decoder_channels[2], decoder_channels[3], act_type=act_type)

        num_classes = decoder_channels[4]
        self.CBR5 = nn.Sequential(
            ConvBNAct(32, 32, 3, padding=1, act_type='relu'),
            nn.Conv2d(32, num_classes, 1)
        )

        self.D_fusion1 = decoder_fusion(32, 16, pool_out[2])
        self.D_fusion2 = decoder_fusion_residual_preserve(64, 32, pool_out[1])
        self.D_fusion3 = decoder_fusion_residual_preserve(32, 16, pool_out[0])

        self.semantic_gate = SemanticGate(
            in_channels=decoder_channels[0],
            x3_channels=decoder_channels[1],
            x1_channels=decoder_channels[3],
            hidden=16
        )

    def forward(self, x1, x3, x4, x5_context):
        g3, g1 = self.semantic_gate(x5_context)

        x3 = x3 * (1 + self.semantic_gate.alpha3 * g3)
        x1 = x1 * (1 + self.semantic_gate.alpha1 * g1)

        x = self.CBR1(x5_context)
        x = F.interpolate(x, size=x4.shape[-2:], mode='bilinear', align_corners=True)
        x = self.D_fusion1(x, x4)

        x = self.CBR2(x)
        x = F.interpolate(x, size=x3.shape[-2:], mode='bilinear', align_corners=True)
        x = self.D_fusion2(x, x3)

        x = self.CBR3(x)
        x = self.CBR4(x)
        x = F.interpolate(x, size=x1.shape[-2:], mode='bilinear', align_corners=True)
        x = self.D_fusion3(x, x1)

        x = self.CBR5(x)
        return x
