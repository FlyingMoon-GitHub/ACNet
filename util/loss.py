# -*- coding: utf-8 -*-

import torch
from torch import nn


class MyLossFunction(object):
    def __init__(self, lambdas):
        self.lambdas = lambdas

    def __call__(self, output, label):
        leaves_out, final_out, final_features = output
        lambda_0, lambda_1, lambda_2 = self.lambdas

        loss = lambda_0 * nn.NLLLoss()(final_out, label)

        for out in leaves_out:
            loss = loss + lambda_1 * nn.NLLLoss()(out, label)

        # Mutual-Channel Diversity Loss

        for f in final_features:
            f_shape = f.shape
            x = f.reshape(f_shape[0], f_shape[1], f_shape[2] * f_shape[3])
            x = nn.Softmax(dim=2)(x)
            x = x.reshape(f_shape[0], f_shape[1], f_shape[2], f_shape[3])

            dc_loss = torch.sum(x, dim=1)
            dc_loss = nn.AdaptiveMaxPool2d((1, 1))(dc_loss)
            dc_loss = torch.mean(dc_loss)

            loss = loss + lambda_2 * dc_loss

        return loss
