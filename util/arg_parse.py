# -*- coding: utf-8 -*-

import argparse
import os
import torch

def getArgs():
    parser = argparse.ArgumentParser()

    '''
    Dataset to be set
    '''
    parser.add_argument('--root', dest='root_path',
                        default=os.path.join('.', 'data'), type=str)
    parser.add_argument('--dataset', dest='dataset',
                        default='CUB-200-2011', type=str)
    parser.add_argument('--img_dir', dest='img_dir',
                        default='images' , type=str)
    parser.add_argument('--anno_dir', dest='anno_dir',
                        default='annos', type=str)

    parser.add_argument('--savepoint_file', dest='savepoint_file',
                        default=None, type=str)
    parser.add_argument('--save_dir', dest='save_dir',
                        default=os.path.join('.', 'checkpoint'), type=str)
    parser.add_argument('--log_dir', dest='log_dir',
                        default=os.path.join('.', 'log'), type=str)
    parser.add_argument('--save_interval', dest='save_interval',
                        default=5, type=int)

    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--aux_conv_in', dest='aux_conv_in',
                        default=0, type=int)
    parser.add_argument('--aux_conv_out', dest='aux_conv_out',
                        default=0, type=int)
    parser.add_argument('--type', dest='type',
                        default='val', type=str)
    parser.add_argument('--class_num', dest='class_num',
                        default=200, type=int)
    parser.add_argument('--target_size', dest='target_size',
                        default=448, type=float)
    parser.add_argument('--tree_height', dest='tree_height',
                        default=3, type=int)

    parser.add_argument('--start_epoch1', dest='start_epoch1',
                        default=0, type=int)
    parser.add_argument('--start_epoch2', dest='start_epoch2',
                        default=0, type=int)
    parser.add_argument('--epoch_num1', dest='epoch_num1',
                        default=60, type=int)
    parser.add_argument('--epoch_num2', dest='epoch_num2',
                        default=200, type=int)
    parser.add_argument('--train_batch', dest='train_batch',
                        default=8, type=int)
    parser.add_argument('--val_batch', dest='val_batch',
                        default=8, type=int)
    parser.add_argument('--test_batch', dest='test_batch',
                        default=8, type=int)
    parser.add_argument('--savepoint', dest='savepoint',
                        default=5000, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        default=5000, type=int)

    parser.add_argument('--lr1', dest='learning_rate1',
                        default=1.0, type=float)
    parser.add_argument('--momentum1', dest='momentum1',
                        default=0.9, type=float)
    parser.add_argument('--decay_step1', dest='decay_step1',
                        default=10, type=int)
    parser.add_argument('--decay_gamma1', dest='decay_gamma1',
                        default=0.25, type=float)
    parser.add_argument('--weight_decay1', dest='weight_decay1',
                        default=5e-6, type=float)

    parser.add_argument('--lr2', dest='learning_rate2',
                        default=0.001, type=float)
    parser.add_argument('--momentum2', dest='momentum2',
                        default=0.9, type=float)
    parser.add_argument('--decay_step2', dest='decay_step2',
                        default=10, type=int)
    parser.add_argument('--decay_gamma2', dest='decay_gamma2',
                        default=0.1, type=float)
    parser.add_argument('--weight_decay2', dest='weight_decay2',
                        default=5e-4, type=float)

    parser.add_argument('--train_num_workers', dest='train_num_workers',
                        default=0, type=int)
    parser.add_argument('--val_num_workers', dest='val_num_workers',
                        default=0, type=int)
    parser.add_argument('--test_num_workers', dest='test_num_workers',
                        default=0, type=int)

    parser.add_argument('--use_cuda', dest='use_cuda',
                        default=torch.cuda.is_available(), type=bool)

    # LACK OF TRANSFORMATION-RELATED ARGUMENTS

    args = parser.parse_args()

    return args