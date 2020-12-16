# -*- coding: utf-8 -*-

import torch
from torch import nn


class MyLossFunction(object):
    def __init__(self, lambdas):
        self.lambdas = lambdas

    def __call__(self, output, label):
        # Cross Entropy Loss

        leaves_out, final_out, final_features = output
        lambda_0, lambda_1, lambda_2, lambda_3, lambda_4 = self.lambdas

        loss = lambda_0 * nn.NLLLoss()(final_out, label)

        for out in leaves_out:
            loss = loss + lambda_1 * nn.NLLLoss()(out, label)

        concat_feature = torch.unsqueeze(final_features[0], dim=1)
        for f in final_features[1:]:
            concat_feature = torch.cat([concat_feature, torch.unsqueeze(f, dim=1)], dim=1)

        f_shape = concat_feature.shape

        # Intra Mutual-Channel Diversity Loss
        if lambda_2:
            x = concat_feature
            x = x.reshape(f_shape[0], f_shape[1], f_shape[2], f_shape[3] * f_shape[4])
            x = nn.Softmax(dim=3)(x)
            x = x.reshape(f_shape[0], f_shape[1], f_shape[2], f_shape[3], f_shape[4])

            dc_loss = torch.sum(x, dim=2)
            dc_loss = nn.AdaptiveMaxPool2d((1, 1))(dc_loss)
            dc_loss = -torch.mean(dc_loss)

            loss = loss + lambda_2 * dc_loss

        # Inter Mutual-Channel Diversity Loss
        if lambda_3:
            x = concat_feature
            x = torch.transpose(x, 1, 2)
            x = x.reshape(f_shape[0], f_shape[2], f_shape[1], f_shape[3] * f_shape[4])
            x = nn.Softmax(dim=3)(x)
            x = x.reshape(f_shape[0], f_shape[2], f_shape[1], f_shape[3], f_shape[4])

            dc_loss = torch.sum(x, dim=2)
            dc_loss = nn.AdaptiveMaxPool2d((1, 1))(dc_loss)
            dc_loss = -torch.mean(dc_loss)

            loss = loss + lambda_3 * dc_loss

        # Ranking Loss

        if lambda_4:
            pass

        return loss
