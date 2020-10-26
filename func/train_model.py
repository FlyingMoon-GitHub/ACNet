# -*- coding: utf-8 -*-

import datetime
import os
import torch
from torch import nn
from torch.autograd import Variable

from func.test_model import *


def train(args, model, optimizers, learning_rate_schedulers, dataloaders):
    loss_records = []

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    train_batch_size = dataloaders['train'].batch_size
    train_epoch_step = len(dataloaders['train'])

    savepoint = args.savepoint
    checkpoint = args.checkpoint
    if savepoint > train_epoch_step:
        savepoint = train_epoch_step
        checkpoint = savepoint

    last_time, cur_time = None, datetime.datetime.now()

    if not args.savepoint_file:

        print('stage 1')

        for epoch in range(args.start_epoch1, args.epoch_num1):

            cur_step = 0

            for batch_no, data in enumerate(dataloaders['train']):

                last_time = cur_time

                cur_step += 1

                model.train(True)

                if args.use_cuda:
                    model.module.backbone.train(False)
                else:
                    model.backbone.train(False)

                image, label = data

                if args.use_cuda:
                    image = Variable(image.cuda())
                    label = Variable(label.cuda()).long()
                else:
                    image = Variable(image)
                    label = Variable(label).long()

                optimizers[0].zero_grad()

                leaves_out, final_out = model(image)

                loss = nn.CrossEntropyLoss()(final_out, label)

                for out in leaves_out:
                    loss = loss + nn.CrossEntropyLoss()(out, label)

                if args.use_cuda:
                    loss = loss.cuda()

                loss.backward()

                if args.use_cuda:
                    torch.cuda.synchronize()

                optimizers[0].step()

                if learning_rate_schedulers[0]:
                    learning_rate_schedulers[0].step()

                if args.use_cuda:
                    torch.cuda.synchronize()

                cur_time = datetime.datetime.now()

                loss_records.append(loss.detach().item())

                print('train_step: {:-8d} / {:d}, loss: {:6.4f}'
                      .format(cur_step, train_epoch_step, loss.detach().item()), flush=True)

                print(cur_time - last_time)

            print('epoch: {:-4d}, start_epoch: {:-4d}, epoch_num: {:-4d}.'
                  .format(epoch, args.start_epoch1, args.epoch_num1))

        if args.use_cuda:
            model.module.backbone.train(False)
        else:
            model.backbone.train(False)

    print('stage 2')

    for epoch in range(args.start_epoch2, args.epoch_num2):

        cur_step = 0

        for batch_no, data in enumerate(dataloaders['train']):

            last_time = cur_time

            cur_step += 1

            model.train(True)

            image, label = data

            if args.use_cuda:
                image = Variable(image.cuda())
                label = Variable(label.cuda()).long()
            else:
                image = Variable(image)
                label = Variable(label).long()

            optimizers[1].zero_grad()

            leaves_out, final_out = model(image)

            loss = nn.CrossEntropyLoss()(final_out, label)

            for out in leaves_out:
                loss = loss + nn.CrossEntropyLoss()(out, label)

            if args.use_cuda:
                loss = loss.cuda()

            loss.backward()

            if args.use_cuda:
                torch.cuda.synchronize()

            optimizers[1].step()

            if learning_rate_schedulers[1]:
                learning_rate_schedulers[1].step()

            if args.use_cuda:
                torch.cuda.synchronize()

            cur_time = datetime.datetime.now()

            loss_records.append(loss.detach().item())

            print('train_step: {:-8d} / {:d}, loss: {:6.4f}'
                  .format(cur_step, train_epoch_step, loss.detach().item()), flush=True)

            print(cur_time - last_time)

        print('epoch: {:-4d}, start_epoch: {:-4d}, epoch_num: {:-4d}.'
              .format(epoch, args.start_epoch2, args.epoch_num2))

        if args.type == 'val':
            val_result = test(args, model=model, dataloader=dataloaders['val'], type='val')
            val_acc = val_result['val_acc']
            print('val_acc: {:6.4f}.'.format(val_acc))

        save_path = os.path.join(args.save_dir, 'weight_epoch_' + str(epoch) + '.pth')

        if args.use_cuda:
            torch.cuda.synchronize()

        torch.save(model.state_dict(), save_path)

        if args.use_cuda:
            torch.cuda.empty_cache()