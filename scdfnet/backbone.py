"""DDRNet backbone extractor.

Splits DDRNet into 4 staged outputs at 1/4, 1/8, 1/16 and 1/32 resolution.
Supported versions: 'ddrnet_23', 'ddrnet_23_slim', 'ddrnet_39'.
"""

import logging

import torch
import torch.nn as nn

from .common import conv3x3, BatchNorm2d, bn_mom


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class DDRNetBackbone(nn.Module):
    """DDRNet backbone extractor.

    Splits DDRNet into 4 staged outputs at 1/4, 1/8, 1/16 and 1/32 resolution.

    Args:
        version: 'ddrnet_23', 'ddrnet_23_slim', or 'ddrnet_39'.
        pretrained: whether to load pretrained weights.
        backbone_path: path to the pretrained backbone weights.
    """

    def __init__(self, version='ddrnet_23', pretrained=False, backbone_path=None):
        super(DDRNetBackbone, self).__init__()

        self.version = version

        if version == 'ddrnet_23':
            layers = [2, 2, 2, 2]
            planes = 64
            self.out_channels = [64, 128, 256, 512]
        elif version == 'ddrnet_23_slim':
            layers = [2, 2, 2, 2]
            planes = 32
            self.out_channels = [32, 64, 128, 256]
        elif version == 'ddrnet_39':
            layers = [3, 4, 6, 3]
            planes = 64
            self.out_channels = [64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported DDRNet version: {version}")

        highres_planes = planes * 2

        # Stem (outputs 1/4)
        self.stem = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=False)

        # Stage 1 (keeps 1/4)
        self.stage1 = self._make_layer(BasicBlock, planes, planes, layers[0])

        # Stage 2 (outputs 1/8)
        self.stage2 = self._make_layer(BasicBlock, planes, planes * 2, layers[1], stride=2)

        # Stage 3 (outputs 1/16)
        if version == 'ddrnet_39':
            # DDRNet-39: Stage 3 is split into two sub-stages.
            self.stage3_1 = self._make_layer(BasicBlock, planes * 2, planes * 4, layers[2] // 2, stride=2)
            self.stage3_2 = self._make_layer(BasicBlock, planes * 4, planes * 4, layers[2] // 2)
            self.stage3 = None
        else:
            self.stage3 = self._make_layer(BasicBlock, planes * 2, planes * 4, layers[2], stride=2)
            self.stage3_1 = None
            self.stage3_2 = None

        # Stage 4 (outputs 1/32)
        self.stage4 = self._make_layer(BasicBlock, planes * 4, planes * 8, layers[3], stride=2)

        # Weight initialisation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrained and backbone_path is not None:
            self._load_pretrained(backbone_path)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def _load_pretrained(self, backbone_path):
        """Load DDRNet pretrained weights with key-name remapping."""
        try:
            checkpoint = torch.load(backbone_path, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Strip possible prefixes (module. / model.)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                if k.startswith('model.'):
                    k = k[6:]
                new_state_dict[k] = v

            # Key mapping: original DDRNet -> our naming
            key_mapping = {
                'conv1': 'stem',
                'layer1': 'stage1',
                'layer2': 'stage2',
                'layer3': 'stage3',
                'layer4': 'stage4',
            }

            if self.version == 'ddrnet_39':
                key_mapping.update({
                    'layer3_1': 'stage3_1',
                    'layer3_2': 'stage3_2',
                })

            mapped_state_dict = {}
            for k, v in new_state_dict.items():
                mapped_key = k
                for old_prefix, new_prefix in key_mapping.items():
                    if k.startswith(old_prefix + '.'):
                        mapped_key = k.replace(old_prefix + '.', new_prefix + '.', 1)
                        break
                mapped_state_dict[mapped_key] = v

            # Keep only backbone weights (exclude classification/segmentation heads)
            model_dict = self.state_dict()
            exclude_prefixes = ['last_layer', 'linear', 'final_layer', 'seghead_extra',
                                'criterion', 'spp', 'layer5', 'layer5_']
            filtered_dict = {k: v for k, v in mapped_state_dict.items()
                             if k in model_dict and not any(k.startswith(prefix) for prefix in exclude_prefixes)}

            loaded_missing, loaded_unexpected = self.load_state_dict(filtered_dict, strict=False)

            if not filtered_dict:
                raise RuntimeError(
                    f"No backbone weights matched the model after key remapping. "
                    f"Check that '{backbone_path}' is a DDRNet checkpoint matching "
                    f"version '{self.version}'.")

            logging.info(f"[DDRNet] Loaded pretrained backbone from: {backbone_path}")
            logging.info(f"[DDRNet] Loaded {len(filtered_dict)} tensors "
                         f"(missing: {len(loaded_missing)}, unexpected: {len(loaded_unexpected)})")

        except Exception as e:
            # Fail loudly: silently training from scratch would look like a
            # reproducibility failure rather than a misconfigured path.
            raise RuntimeError(
                f"Failed to load pretrained backbone from '{backbone_path}': {e}") from e

    def forward(self, x):
        """Forward pass returning 4-scale features (pure staged backbone).

        Returns:
            x1: 1/4 scale  [B, C1, H/4, W/4]
            x2: 1/8 scale  [B, C2, H/8, W/8]
            x3: 1/16 scale [B, C3, H/16, W/16]
            x4: 1/32 scale [B, C4, H/32, W/32]
        """
        x = self.stem(x)
        x1 = self.stage1(x)
        x2 = self.stage2(self.relu(x1))

        if self.version == 'ddrnet_39':
            x = self.stage3_1(self.relu(x2))
            x3 = self.stage3_2(self.relu(x))
        else:
            x3 = self.stage3(self.relu(x2))

        x4 = self.stage4(self.relu(x3))
        return x1, x2, x3, x4


def create_ddrnet_backbone(version='ddrnet_23', pretrained=False, backbone_path=None):
    """Factory function to create a DDRNet backbone."""
    return DDRNetBackbone(version=version, pretrained=pretrained, backbone_path=backbone_path)
