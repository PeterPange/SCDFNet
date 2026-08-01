"""SCDFNet: Saliency-Calibrated Discrepancy Fusion Network for Real-Time
Multimodal (RGB-X) Semantic Segmentation.

This package contains the full model definition of SCDFNet, with three
backbone scales: SCDFNet-1-slim (DDRNet-23-slim), SCDFNet-1 (DDRNet-23) and
SCDFNet-2 (DDRNet-39). Module names follow the paper:

  - CSPC: Consistency-Saliency Pre-fusion Calibration (applied before both
    fusion stages). It contains CMCA (Cross-Modal Consistency Alignment) and
    SCDS (Saliency-Calibrated Discrepancy Suppression) plus a weak residual
    update with a learnable scalar alpha.
  - DGRF: Difference-Guided Residual Fusion (the second-stage fusion).
    SC = rgb + Up(beta) * x, where beta is a difference-aware gate.
  - CSDB: Cross-Scale Detail Bridge, injecting 1/4 detail (x1) into the 1/8
    fused feature with a relation-aware gate and a learnable scalar gamma.
  - DAPPM3: 3-branch progressive aggregation context module.
  - Decoder: a lightweight decoder for the final segmentation output.

Package layout:
    common.py    shared blocks (ConvBNAct, fusion_attention, pool configs)
    backbone.py  DDRNet backbone (BasicBlock, Bottleneck, DDRNetBackbone)
    cmca.py      Cross-Modal Consistency Alignment
    scds.py      Saliency-Calibrated Discrepancy Suppression
    cspc.py      Consistency-Saliency Pre-fusion Calibration (CMCA + SCDS)
    fusion.py    encoder_fusion (Fusion1) + DGRF (Fusion2)
    csdb.py      Cross-Scale Detail Bridge
    xguide.py    X-branch guidance gate
    dappm.py     DAPPM3 context module
    decoder.py   Decoder
    encoder.py   SCDFNetEncoder
    network.py   SCDFNet + build_scdfnet factory
"""

from .common import (
    ConvBNAct,
    fusion_attention,
    conv3x3,
    _get_pool_config,
    BatchNorm2d,
    bn_mom,
)
from .backbone import (
    BasicBlock,
    Bottleneck,
    DDRNetBackbone,
    create_ddrnet_backbone,
)
from .cmca import CMCA
from .scds import SFCBranch, Fuse_sfc_dual, SCDS
from .cspc import CSPC
from .fusion import encoder_fusion, DGRFGate, DGRF
from .csdb import CSDB
from .xguide import XGuide16
from .dappm import DAPPM3
from .decoder import (
    decoder_fusion,
    decoder_fusion_residual_preserve,
    SemanticGate,
    Decoder,
)
from .encoder import SCDFNetEncoder
from .network import SCDFNet, build_scdfnet

__all__ = [
    "build_scdfnet",
    "SCDFNet",
    "SCDFNetEncoder",
    "CMCA",
    "SCDS",
    "CSPC",
    "DGRF",
    "DGRFGate",
    "CSDB",
    "XGuide16",
    "DAPPM3",
    "SemanticGate",
    "Decoder",
    "encoder_fusion",
    "fusion_attention",
    "decoder_fusion",
    "decoder_fusion_residual_preserve",
    "SFCBranch",
    "Fuse_sfc_dual",
    "ConvBNAct",
    "DDRNetBackbone",
    "BasicBlock",
    "Bottleneck",
    "create_ddrnet_backbone",
    "conv3x3",
    "_get_pool_config",
    "BatchNorm2d",
    "bn_mom",
]
