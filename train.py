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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    args = getArgs()

    # CUSTOM SETTINGS
    args.save_dir = os.path.join('.', 'checkpoint', args.dataset)
    args.log_dir = os.path.join('.', 'log', args.dataset)
    # CUSTOM SETTINGS END

    args.use_cuda = args.use_cuda and torch.cuda.is_available()

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
        model.tree.apply(weightInit)

    if args.use_cuda:
        model = model.cuda()

    # model.summary()
    model.saveGraph()

    if args.use_cuda:
        model = nn.DataParallel(model)

    optimizer1 = None
    if args.use_cuda:
        optimizer1 = optim.SGD(model.module.tree.parameters(), lr=args.learning_rate1,
                               momentum=args.momentum1, weight_decay=args.weight_decay1)
    else:
        optimizer1 = optim.SGD(model.tree.parameters(), lr=args.learning_rate1,
                               momentum=args.momentum1, weight_decay=args.weight_decay1)

    learning_rate_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=args.decay_step1, gamma=args.decay_gamma1)

    optimizer2 = None
    if args.use_cuda:
        optimizer2 = optim.SGD(model.module.parameters(), lr=args.learning_rate2,
                               momentum=args.momentum2, weight_decay=args.weight_decay2)
    else:
        optimizer2 = optim.SGD(model.parameters(), lr=args.learning_rate2,
                               momentum=args.momentum2, weight_decay=args.weight_decay2)

    learning_rate_scheduler2 = lr_scheduler.StepLR(optimizer2, step_size=args.decay_step2, gamma=args.decay_gamma2)

    train(args=args,
          model=model,
          optimizers=[optimizer1, optimizer2],
          learning_rate_schedulers=[learning_rate_scheduler1, learning_rate_scheduler2],
          dataloaders=dataloaders)
