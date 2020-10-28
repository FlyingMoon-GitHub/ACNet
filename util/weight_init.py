# -*- coding: utf-8 -*-

from torch import nn

def weightInit(module):
    if isinstance(module, nn.Conv2d):
        if module.weight is not None:
            nn.init.normal_(module.weight.data, 0, 0.05)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.Linear):
        if module.weight is not None:
            nn.init.xavier_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm2d):
        if module.weight is not None:
            module.weight.data.fill_(1)
        if module.bias is not None:
            module.bias.data.zero_()
