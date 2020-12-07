# -*- coding: utf-8 -*-

from torch import nn


class MyLossFunction(object):
    def __init__(self):
        pass

    def __call__(self, leaves_out, final_out, label, final_features=()):
        loss = nn.NLLLoss()(final_out, label)

        for out in leaves_out:
            loss = loss + nn.NLLLoss()(out, label)

        return loss
