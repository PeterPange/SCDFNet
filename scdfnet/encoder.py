"""SCDFNet encoder.

Pipeline:
  - DDRNet RGB backbone + shared X branch (stem/stage1/stage2).
  - CSPC before Fusion1 and before DGRF (Fusion2).
  - Fusion1 (cosine attention) producing the 1/4 detail x1.
  - DGRF (difference-guided residual fusion) at 1/8.
  - CSDB injecting the 1/4 detail into the 1/8 fused feature.
  - An X-branch gate modulating the 1/16 feature with the raw X 1/8 feature.
"""

import copy

import torch
import torch.nn as nn

from .common import ConvBNAct, _get_pool_config
from .backbone import create_ddrnet_backbone
from .cspc import CSPC
from .fusion import encoder_fusion, DGRF
from .csdb import CSDB
from .xguide import XGuide16


class SCDFNetEncoder(nn.Module):
    """SCDFNet encoder.

    Pipeline:
      - DDRNet RGB backbone + shared X branch (stem/stage1/stage2).
      - CSPC before Fusion1 and before DGRF (Fusion2).
      - Fusion1 (cosine attention) producing the 1/4 detail x1.
      - DGRF (difference-guided residual fusion) at 1/8.
      - CSDB injecting the 1/4 detail into the 1/8 fused feature.
      - An X-branch gate modulating the 1/16 feature with the raw X 1/8 feature.
    """

    def __init__(self, backbone='ddrnet_23', pretrained=False, backbone_path=None,
                 dataset="MFNet", pool_out=None):
        super().__init__()

        self.dataset = dataset

        if pool_out is None:
            pool_out, _ = _get_pool_config(dataset)

        # RGB backbone
        self.rgb_backbone = create_ddrnet_backbone(backbone, pretrained, backbone_path)

        if backbone == 'ddrnet_23_slim':
            out_channels = [32, 64, 128, 256]
        elif backbone == 'ddrnet_23':
            out_channels = [64, 128, 256, 512]
        elif backbone == 'ddrnet_39':
            out_channels = [64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.out_channels = out_channels

        # X branch
        if dataset == "Cityscapes":
            x_in_channels = 2
        elif dataset == "ZJU":
            x_in_channels = 3
        else:
            x_in_channels = 1

        x_out_channels = out_channels[0]
        self.x_stem = nn.Sequential(
            nn.Conv2d(x_in_channels, x_out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(x_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(x_out_channels, x_out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(x_out_channels),
            nn.ReLU(inplace=True)
        )

        self.x_stage1 = copy.deepcopy(self.rgb_backbone.stage1)
        self.x_stage2 = copy.deepcopy(self.rgb_backbone.stage2)

        # Agent layers
        self.agent1 = ConvBNAct(out_channels[0], 32, kernel_size=1, padding=0, act_type='relu')
        self.agent3 = ConvBNAct(out_channels[1], 64, kernel_size=1, padding=0, act_type='relu')
        self.agent4 = ConvBNAct(out_channels[2], 32, kernel_size=1, padding=0, act_type='relu')

        # CSPC before Fusion1
        self.cspc_f1 = CSPC(
            dim=out_channels[0], lambda_init=0.1, alpha_init=0.05
        )

        # Fusion1 (cosine attention, unchanged)
        self.fusion1 = encoder_fusion(
            out_channels[0], out_channels[0] // 2, pool_out[0],
            skip_conncetion=True, last_fusion=False
        )

        # CSPC before DGRF (Fusion2)
        self.cspc_f2 = CSPC(
            dim=out_channels[1], lambda_init=0.2, alpha_init=0.1
        )

        # DGRF (Fusion2)
        self.dgrf = DGRF(
            out_channels[1], 32, pool_out[1]
        )

        # CSDB: bridge x1 detail into s3
        self.csdb = CSDB(
            in_low=32,
            in_high=out_channels[1],
            out_high=out_channels[1],
            gamma_init=0.1
        )

        # X-branch guidance gate @ 1/16 (uses raw x_s2)
        self.x_guide16 = XGuide16(
            in_channels=out_channels[1],
            out_channels=out_channels[2]
        )

        # Initialise X branch from RGB
        self._init_x_from_rgb()

    def _init_x_from_rgb(self):
        """Initialise the X branch from the RGB branch."""
        rgb_stem_conv1 = self.rgb_backbone.stem[0]
        x_stem_conv1 = self.x_stem[0]

        if x_stem_conv1.in_channels == 3:
            x_weight = rgb_stem_conv1.weight.data.clone()
            x_stem_conv1.weight.data.copy_(x_weight)
        elif x_stem_conv1.in_channels == 1:
            x_weight = rgb_stem_conv1.weight.data.mean(dim=1, keepdim=True)
            x_stem_conv1.weight.data.copy_(x_weight)
        elif x_stem_conv1.in_channels == 2:
            mean_weight = rgb_stem_conv1.weight.data.mean(dim=1, keepdim=True)
            x_weight = mean_weight.repeat(1, 2, 1, 1)
            x_stem_conv1.weight.data.copy_(x_weight)

        rgb_stem_conv2 = self.rgb_backbone.stem[3]
        x_stem_conv2 = self.x_stem[3]
        x_stem_conv2.weight.data.copy_(rgb_stem_conv2.weight.data)

        for param_name in ['weight', 'bias', 'running_mean', 'running_var']:
            getattr(self.x_stem[1], param_name).data.copy_(getattr(self.rgb_backbone.stem[1], param_name).data)
        for param_name in ['weight', 'bias', 'running_mean', 'running_var']:
            getattr(self.x_stem[4], param_name).data.copy_(getattr(self.rgb_backbone.stem[4], param_name).data)

        self.x_stage1.load_state_dict(self.rgb_backbone.stage1.state_dict())
        self.x_stage2.load_state_dict(self.rgb_backbone.stage2.state_dict())

    def forward(self, rgb, x):
        """Forward pass.

        Returns:
            x1: 1/4 scale  [B, 32, H/4, W/4]
            x3: 1/8 scale  [B, 64, H/8, W/8]
            x4: 1/16 scale [B, 32, H/16, W/16]
            x5: 1/32 scale [B, C4, H/32, W/32]
        """
        # Stage 1 (1/4)
        rgb_s1_raw = self.rgb_backbone.stem(rgb)
        rgb_s1_raw = self.rgb_backbone.stage1(rgb_s1_raw)

        x_s1_raw = self.x_stem(x)
        x_s1_raw = self.x_stage1(x_s1_raw)

        # CSPC (CMCA then SCDS, weak residual) before Fusion1
        rgb_s1, x_s1 = self.cspc_f1(rgb_s1_raw, x_s1_raw)

        rgb_s1, x_s1, s1 = self.fusion1(rgb_s1, x_s1)

        x1 = self.agent1(s1)  # [B, 32, H/4, W/4]

        # Stage 2 (1/8)
        rgb_s2_raw = self.rgb_backbone.stage2(self.rgb_backbone.relu(rgb_s1))
        x_s2_raw = self.x_stage2(self.rgb_backbone.relu(x_s1))

        # CSPC before DGRF
        rgb_s2, x_s2 = self.cspc_f2(rgb_s2_raw, x_s2_raw)

        s3 = self.dgrf(rgb_s2, x_s2)

        # Bridge x1 detail into s3
        s3 = self.csdb(x1, s3)

        x3 = self.agent3(s3)  # [B, 64, H/8, W/8]

        # Stage 3 (1/16) - X-branch guidance gate (uses raw x_s2)
        if self.rgb_backbone.version == 'ddrnet_39':
            s4 = self.rgb_backbone.stage3_1(self.rgb_backbone.relu(s3))
            s4 = self.rgb_backbone.stage3_2(self.rgb_backbone.relu(s4))
        else:
            s4 = self.rgb_backbone.stage3(self.rgb_backbone.relu(s3))

        x_guide16, gate16 = self.x_guide16(x_s2_raw)
        s4 = s4 * (1 + gate16)

        x4 = self.agent4(s4)  # [B, 32, H/16, W/16]

        # Stage 4 (1/32) - RGB branch only
        x5 = self.rgb_backbone.stage4(self.rgb_backbone.relu(s4))

        return x1, x3, x4, x5
