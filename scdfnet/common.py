"""Shared building blocks used across SCDFNet modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F


BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def _get_pool_config(dataset):
    """Return pooling configuration for a given dataset.

    Returns:
        (pool_out, context_pool): pool_out is the 4-stage fusion-attention pool
        sizes, context_pool is the DAPPM3 mid-scale pool size.
    """
    if dataset == "Cityscapes":
        return [(16, 32), (8, 16), (4, 8), (2, 4)], (4, 8)
    if dataset in ["MFNet", "ZJU", "FMB"]:
        return [(16, 24), (8, 12), (4, 6), (2, 3)], (5, 5)
    else:
        return [(8, 8), (4, 4), (2, 2), (1, 1)], (2, 2)


class ConvBNAct(nn.Module):
    """Conv + BN + Activation."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 bias=False, act_type='relu', **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

        activation_hub = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU,
                          'prelu': nn.PReLU, 'elu': nn.ELU,
                          'selu': nn.SELU, 'silu': nn.SiLU,
                          'sigmoid': nn.Sigmoid, 'softmax': nn.Softmax,
                          'logsoftmax': nn.LogSoftmax, 'tanh': nn.Tanh}

        assert act_type in activation_hub.keys(), f"act_type must be one of {activation_hub.keys()}"
        self.act = activation_hub[act_type](**kwargs)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class fusion_attention(nn.Module):
    """Cosine-similarity based fusion attention.

    Shared by the encoder Fusion1 module and the decoder fusion modules.
    """

    def __init__(self, num_channel, out_channel, pool_size):
        super(fusion_attention, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

        self.conv1 = nn.Conv2d(num_channel, out_channel, kernel_size=1)
        self.conv2 = nn.Conv2d(out_channel, num_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.activation = nn.Sigmoid()

    def forward(self, rgb, depth):
        rgb = self.pool(rgb).view((rgb.size()[0], rgb.size()[1], -1))
        depth = self.pool(depth).view((depth.size()[0], depth.size()[1], -1))
        similarity_vector = self.cos(rgb, depth).view(rgb.size()[0], rgb.size()[1], 1, 1)

        output = F.relu(self.bn1(self.conv1(similarity_vector)))
        output = self.conv2(output)
        weight = self.activation(output)
        return weight, 1 - weight
