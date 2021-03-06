# -*- coding: utf-8 -*-

import os
import torch
from torch.utils.data import DataLoader

from data.dataset import *
from func.train_model import *
from model.acnet import *
from util.arg_parse import *
from util.config import *
from util.lr_lambda import *
from util.weight_init import *

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    args = getArgs()

    assert args.type in ['train', 'val']

    dataloaders = {}

    train_data_config = getDatasetConfig(args, 'train')
    train_dataset = MyDataset(train_data_config)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.train_batch,
                                  shuffle=True,
                                  num_workers=args.train_num_workers,
                                  drop_last=False,
                                  pin_memory=True)
    dataloaders['train'] = train_dataloader

    if args.type == 'val':
        val_data_config = getDatasetConfig(args, 'val')
        val_dataset = MyDataset(val_data_config)
        val_dataloader = DataLoader(dataset=val_dataset,
                                    batch_size=args.val_batch,
                                    shuffle=True,
                                    num_workers=args.val_num_workers,
                                    drop_last=False,
                                    pin_memory=True)
        dataloaders['val'] = val_dataloader

    model_config = getModelConfig(args, args.type)

    model = ACNet(model_config)

    if args.savepoint_file:
        model_dict = model.state_dict()
        model_dict.update({k.replace('module.', ''): v for k, v in torch.load(args.savepoint_file).items()})
        model.load_state_dict(model_dict)
    else:
        model.aux_conv.apply(weightInit)
        model.tree.apply(weightInit)

    if args.use_cuda:
        model = model.cuda()

    if args.summary:
        model.summary()
    if args.save_graph:
        model.saveGraph()

    if args.use_cuda:
        model = nn.DataParallel(model)
        if torch.cuda.device_count() > 1:
            model = model.to(torch.device('cuda:0'))

    optimizer1 = None
    if args.use_cuda:
        optimizer1 = optim.SGD([*model.module.aux_conv.parameters(), *model.module.tree.parameters()], lr=args.learning_rate1,
                               momentum=args.momentum1, weight_decay=args.weight_decay1)
    else:
        optimizer1 = optim.SGD([*model.aux_conv.parameters(), *model.tree.parameters()], lr=args.learning_rate1,
                               momentum=args.momentum1, weight_decay=args.weight_decay1)

    learning_rate_scheduler1 = lr_scheduler.LambdaLR(optimizer1, lr_lambda=lambda x: lr_lambda1(x, args))

    optimizer2 = None
    if args.use_cuda:
        optimizer2 = optim.SGD(model.module.parameters(), lr=args.learning_rate2,
                               momentum=args.momentum2, weight_decay=args.weight_decay2)
    else:
        optimizer2 = optim.SGD(model.parameters(), lr=args.learning_rate2,
                               momentum=args.momentum2, weight_decay=args.weight_decay2)

    learning_rate_scheduler2 = lr_scheduler.LambdaLR(optimizer2, lr_lambda=lambda x: lr_lambda2(x, args))

    train(args=args,
          model=model,
          optimizers=[optimizer1, optimizer2],
          learning_rate_schedulers=[learning_rate_scheduler1, learning_rate_scheduler2],
          dataloaders=dataloaders)
