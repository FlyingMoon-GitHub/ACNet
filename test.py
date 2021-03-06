# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import DataLoader

from data.dataset import *
from func.train_model import *
from model.acnet import *
from util.arg_parse import *
from util.config import *
from util.weight_init import *

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    args = getArgs()

    assert args.type in ['test']

    test_data_config = getDatasetConfig(args, 'test')
    test_dataset = MyDataset(test_data_config)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=args.test_batch,
                                 shuffle=True,
                                 num_workers=args.test_num_workers,
                                 drop_last=False,
                                 pin_memory=True)

    model_config = getModelConfig(args, 'test')

    model = ACNet(model_config)

    if args.savepoint_file:
        model_dict = model.state_dict()
        model_dict.update({k.replace('module.', ''): v for k, v in torch.load(args.savepoint_file).items()})
        model.load_state_dict(model_dict)
    else:
        model.apply(weightInit)

    if args.use_cuda:
        model = model.cuda()

    if args.summary:
        model.summary()

    if args.use_cuda:
        model = nn.DataParallel(model)
        if torch.cuda.device_count() > 1:
            model = model.to(torch.device('cuda:0'))

    test_result = test(args, model=model, dataloader=test_dataloader, type='test')

    test_acc = test_result['test_acc']
    test_cfs_mat = test_result['test_cfs_mat']
    print('test_acc: {:6.4f}.'.format(test_acc))
    print('test_cfs_mat')
    print(test_cfs_mat)
