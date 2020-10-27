# -*- coding: utf-8 -*-

from torch import nn
from torchvision.models import *

from util.weight_init import *

def getBackbone(backbone_name, pretrained=True):
    base_backbone = eval(backbone_name)(pretrained=pretrained)
    all_base_modules = [m for m in base_backbone.children()]

    assert backbone_name in ['vgg16', 'vgg19', 'resnet50', 'resnet101']

    backbone = None

    if backbone_name == 'vgg16':
        feature_extractor = all_base_modules[0]
        backbone = feature_extractor[0:30]
    elif backbone_name == 'vgg19':
        feature_extractor = all_base_modules[0]
        backbone = feature_extractor[0:35]
    elif backbone_name in ['resnet50', 'resnet101']:
        layers = all_base_modules[:-3]
        extra_conv = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        extra_conv.apply(weightInit)
        layers.append(extra_conv)
        backbone = nn.Sequential(*layers)

    return backbone
