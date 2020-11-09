# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F

from model.gcblock import *


class BranchRoutingModule(nn.Module):
    def __init__(self, in_channels=512, epsilon=1e-12):
        super(BranchRoutingModule, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gcblock = GCBlock(in_channels, in_channels)
        self.l2norm = lambda x: F.normalize(x, dim=1)
        self.fc = nn.Linear(in_channels, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.signedsqrt = lambda x: torch.sign(x) * torch.sqrt(torch.sign(x) * x + epsilon)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        feature = self.conv1(x)
        feature = self.gcblock(feature)
        feature = self.avgpool(feature)
        feature = self.signedsqrt(feature)
        feature = self.l2norm(feature)

        feature = feature.view(feature.size(0), -1)

        out = self.fc(feature)
        out = self.sigmoid(out)

        return out
