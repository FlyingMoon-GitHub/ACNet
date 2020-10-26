# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F


class LabelPredictionModule(nn.Module):
    def __init__(self, class_num, in_channels=512):
        super(LabelPredictionModule, self).__init__()
        self.class_num = class_num

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.l2norm = nn.LayerNorm((in_channels, 1, 1))
        self.fc = nn.Linear(in_channels, self.class_num)
        self.softmax = nn.Softmax(dim=1)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.signedsqrt = lambda x: torch.sign(x) * torch.sqrt(torch.sign(x) * x)

    def forward(self, x):
        feature = self.bn(x)
        feature = self.conv1(feature)
        feature = self.maxpool(feature)
        feature = self.signedsqrt(feature)
        feature = self.l2norm(feature)

        feature = feature.view(feature.size(0), -1)

        out = self.fc(feature)
        out = self.softmax(out)

        return out
