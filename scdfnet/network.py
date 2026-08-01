"""SCDFNet full network + factory function.

Composes SCDFNetEncoder + DAPPM3 context + Decoder into the full
multimodal segmentation network, and provides the ``build_scdfnet`` factory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import _get_pool_config
from .dappm import DAPPM3
from .decoder import Decoder
from .encoder import SCDFNetEncoder


class SCDFNet(nn.Module):
    """SCDFNet multimodal segmentation network.

    - Encoder: SCDFNetEncoder (CSPC + Fusion1 + DGRF + CSDB)
    - Context: DAPPM3 (3-branch progressive aggregation)
    - Decoder: lightweight segmentation decoder
    """

    def __init__(self, version='SCDFNet-2', pretrain=False, backbone_path=None,
                 dataset="MFNet", num_class=14, act_type='relu'):
        super().__init__()

        # Parse backbone version from the SCDFNet variant name.
        version_lower = version.lower()
        if 'slim' in version_lower:
            backbone = 'ddrnet_23_slim'
        elif 'scdfnet-2' in version_lower or 'ddrnet_39' in version_lower or 'ddrnet-39' in version_lower:
            backbone = 'ddrnet_39'
        else:
            backbone = 'ddrnet_23'

        self.encoder = SCDFNetEncoder(
            backbone=backbone, pretrained=pretrain, backbone_path=backbone_path,
            dataset=dataset
        )

        out_channels = self.encoder.out_channels
        decoder_channels = [32, 64, 128, 32, num_class]

        # Context module: DAPPM3
        pool_out, context_pool = _get_pool_config(dataset)
        self.dappm3 = DAPPM3(
            in_channels=out_channels[3],
            out_channels=decoder_channels[0],
            branch_channels=64,
            pool_mid=context_pool
        )

        # Segmentation decoder
        self.decoder = Decoder(
            decoder_channels=decoder_channels,
            act_type=act_type,
            pool_out=pool_out,
            dataset=dataset
        )

    def forward(self, rgb, x):
        original_h, original_w = rgb.shape[2], rgb.shape[3]

        x1, x3, x4, x5 = self.encoder(rgb, x)
        x5_context = self.dappm3(x5)
        out_1_4 = self.decoder(x1, x3, x4, x5_context)
        out = F.interpolate(out_1_4, size=(original_h, original_w), mode='bilinear', align_corners=True)

        return out


def build_scdfnet(version='SCDFNet-2', pretrain=False, backbone_path=None,
                  dataset="MFNet", num_classes=14):
    """Factory function that builds the SCDFNet model.

    Args:
        version: SCDFNet variant name. One of:
            'SCDFNet-1-slim' (DDRNet-23-slim),
            'SCDFNet-1'      (DDRNet-23),
            'SCDFNet-2'      (DDRNet-39, default).
        pretrain: whether to load pretrained backbone weights.
        backbone_path: path to pretrained backbone weights (if pretrain=True).
        dataset: dataset name (used for input-channel and pool-size config).
        num_classes: number of segmentation classes.
    """
    return SCDFNet(
        version=version, pretrain=pretrain, backbone_path=backbone_path,
        dataset=dataset, num_class=num_classes
    )
