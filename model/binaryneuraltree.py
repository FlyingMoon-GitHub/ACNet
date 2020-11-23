# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F

from model.attentiontransformer import *
from model.branchrouting import *
from model.labelprediction import *


class BinaryNeuralTree(nn.Module):
    def __init__(self, class_num, tree_height=3, in_channels=512):
        super(BinaryNeuralTree, self).__init__()

        assert tree_height > 1

        self.class_num = class_num
        self.tree_height = tree_height
        self.in_channels = in_channels

        self.branch_routings = self._branch_routings()
        self.attention_transformers = self._attention_transformers()
        self.label_predictions = self._label_predictions()
        self.logs = self._log()

    def _branch_routings(self):
        structure = [[None for _ in range(int(pow(2, i)))] for i in range(self.tree_height - 1)]
        cur = 0
        for i in range(self.tree_height - 1):
            for j in range(int(pow(2, i))):
                self.__setattr__('branch_routing_module' + str(cur), BranchRoutingModule(in_channels=self.in_channels))
                structure[i][j] = self.__getattr__('branch_routing_module' + str(cur))
                cur += 1

        return structure

    def _attention_transformers(self):
        structure = [[None for _ in range(int(pow(2, i + 1)))] for i in range(self.tree_height - 1)]
        cur = 0
        for i in range(self.tree_height - 1):
            for j in range(int(pow(2, i + 1))):
                if j % 2:
                    self.__setattr__('attention_transformer' + str(cur), nn.Sequential(*[AttentionTransformer()]))
                else:
                    self.__setattr__('attention_transformer' + str(cur),
                                     nn.Sequential(*[AttentionTransformer(), AttentionTransformer()]))
                structure[i][j] = self.__getattr__('attention_transformer' + str(cur))
                cur += 1

        return structure

    def _label_predictions(self):
        structure = [None] * int(pow(2, self.tree_height - 1))
        cur = 0
        for i in range(int(pow(2, self.tree_height - 1))):
            self.__setattr__('label_prediction_module' + str(cur), LabelPredictionModule(class_num=self.class_num))
            structure[i] = self.__getattr__('label_prediction_module' + str(cur))
            cur += 1

        return structure

    def _log(self, epsilon=1e-12):
        structure = [None] * int(pow(2, self.tree_height - 1) + 1)
        cur = 0
        for i in range(int(pow(2, self.tree_height - 1)) + 1):
            self.__setattr__('log' + str(cur), (lambda x: torch.log(x + epsilon)))
            structure[i] = self.__getattribute__('log' + str(cur))
            cur += 1

        return structure

    def forward(self, x):
        probs = [[None for _ in range(int(pow(2, i + 1)))] for i in range(self.tree_height - 1)]
        features = [[None for _ in range(int(pow(2, i)))] for i in range(self.tree_height)]

        features[0][0] = x

        for i in range(self.tree_height - 1):
            for j in range(int(pow(2, i))):
                temp_prob = self.branch_routings[i][j](features[i][j])
                probs[i][j * 2] = 1 - temp_prob
                probs[i][j * 2 + 1] = temp_prob
                if i:
                    probs[i][j * 2] = probs[i][j * 2] * probs[i - 1][j]
                    probs[i][j * 2 + 1] = probs[i][j * 2 + 1] * probs[i - 1][j]

                features[i + 1][j * 2] = self.attention_transformers[i][j * 2](features[i][j])
                features[i + 1][j * 2 + 1] = self.attention_transformers[i][j * 2 + 1](features[i][j])

        leaves_out = [self.label_predictions[i](features[self.tree_height - 1][i]) for i in
                      range(int(pow(2, self.tree_height - 1)))]

        final_out = probs[self.tree_height - 2][0] * leaves_out[0]
        for i in range(1, int(pow(2, self.tree_height - 1))):
            final_out = final_out + probs[self.tree_height - 2][i] * leaves_out[i]

        leaves_out = tuple((self.logs[i](leaves_out[i]) for i in range(int(pow(2, self.tree_height - 1)))))

        final_out = self.logs[int(pow(2, self.tree_height - 1))](final_out)

        return leaves_out, final_out
