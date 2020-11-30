# -*- coding: utf-8 -*-

import argparse
import torch.optim as optim
from torch.optim import lr_scheduler

from util.transformation import *

def getDatasetConfig(args, type):
    assert type in ['train', 'val', 'test']

    config = {}

    config['root_path'] = args.root_path
    config['dataset'] = args.dataset
    config['img_dir'] = args.img_dir
    config['anno_dir'] = args.anno_dir

    config['class_num'] = args.class_num
    if type == 'train':
        config['anno_file'] = 'annos_train.txt'
    elif type in ['val', 'test']:
        config['anno_file'] = 'annos_test.txt'

    config['img_aug'] = getTransformation(args, type)

    return config

def getModelConfig(args, type):
    assert type in ['train', 'val', 'test']

    config = {}

    config['backbone'] = args.backbone
    config['aux_conv_in'] = args.aux_conv_in
    config['tree_in'] = args.tree_in
    config['pretrained'] = (args.savepoint_file is None)
    config['class_num'] = args.class_num
    config['target_size'] = args.target_size
    config['tree_height'] = args.tree_height

    config['use_cuda'] = args.use_cuda
    config['log_dir'] = args.log_dir

    return config