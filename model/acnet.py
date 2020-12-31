# -*- coding: utf-8 -*-

import torch
from torch import nn
from torchsummary import *
from tensorboardX import SummaryWriter

from model.backbone import *
from model.binaryneuraltree import *


class ACNet(nn.Module):

    def __init__(self, config):
        super(ACNet, self).__init__()

        self.target_size = (3, config['target_size'], config['target_size'])
        self.class_num = config['class_num']
        self.aux_conv_in = config['aux_conv_in']
        self.tree_in = config['tree_in']
        self.use_cuda = config['use_cuda']
        self.log_dir = config['log_dir']
        self.tree_height = config['tree_height']

        self.pretrained = config['pretrained']

        self.backbone = getBackbone(config['backbone'], pretrained=self.pretrained)
        self.aux_conv = self._aux_conv(self.aux_conv_in, self.tree_in)
        self.tree = BinaryNeuralTree(class_num=self.class_num, tree_height=self.tree_height, in_channels=self.tree_in)

    def _aux_conv(self, aux_conv_in, aux_conv_out):
        layers = []
        if aux_conv_in > 0:
            extra_conv = nn.Conv2d(in_channels=aux_conv_in, out_channels=aux_conv_out, kernel_size=1)
            extra_conv.apply(weightInit)
            layers.append(extra_conv)
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        init_feature = self.backbone(x)
        init_feature = self.aux_conv(init_feature)

        leaves_out, final_out, final_features, penultimate_out = self.tree(init_feature)

        return leaves_out, final_out, final_features, penultimate_out

    def summary(self):
        summary(self, [self.target_size], device="cuda" if self.use_cuda else "cpu")

    def saveGraph(self):
        with SummaryWriter(comment='acnet', log_dir=self.log_dir) as sumw:
            input_tensor = torch.rand((1,) + self.target_size)
            if self.use_cuda:
                input_tensor = input_tensor.cuda()
            sumw.add_graph(self, input_tensor)


if __name__ == '__main__':
    from util.arg_parse import *
    from util.config import *

    args = getArgs()

    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    model_config = getModelConfig(args, 'train')

    model = ACNet(model_config)

    if args.use_cuda:
        model = model.cuda()

    # model.summary()
    model.saveGraph()
