# -*- coding: utf-8 -*-

import datetime
import os
import torch
from torch import nn
from torch.autograd import Variable

from func.test_model import *

from util.loss import *


def train(args, model, optimizers, learning_rate_schedulers, dataloaders):
    loss_records = []

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.log_file:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        with open(os.path.join(args.log_dir, args.log_file), 'a+') as log_file:
            log_file.write(('-' * 5) + '\n')


    train_batch_size = dataloaders['train'].batch_size
    train_epoch_step = len(dataloaders['train'])

    savepoint = args.savepoint
    checkpoint = args.checkpoint
    if savepoint > train_epoch_step:
        savepoint = train_epoch_step
        checkpoint = savepoint

    lambdas = (args.lambda_0, args.lambda_1, args.lambda_2, args.lambda_3, args.lambda_4)
    margin = args.margin
    loss_func = MyLossFunction(lambdas, margin)

    last_time, cur_time = None, datetime.datetime.now()

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

            leaves_out, final_out, final_features, penultimate_out = model(image)

            target = torch.ones([leaves_out[0].shape[0]])

            if args.use_cuda:
                target = target.cuda()

            loss = loss_func((leaves_out, final_out, final_features, penultimate_out), label, target)

            if args.use_cuda:
                loss = loss.cuda()

            loss.backward()

            if args.use_cuda:
                torch.cuda.synchronize()

            optimizers[0].step()

            if args.use_cuda:
                torch.cuda.synchronize()

            cur_time = datetime.datetime.now()

            loss_records.append(loss.detach().item())

            print('train_step: {:-8d} / {:d}, loss: {:6.4f}'
                  .format(cur_step, train_epoch_step, loss.detach().item()), flush=True)

            print(cur_time - last_time)

        if learning_rate_schedulers[0]:
            learning_rate_schedulers[0].step()

        print('stage 1')
        print('epoch: {:-4d}, start_epoch: {:-4d}, epoch_num: {:-4d}.'
              .format(epoch, args.start_epoch1, args.epoch_num1))

        if args.type == 'val':
            if (epoch + 1) % args.val_interval == 0:
                val_result = test(args, model=model, dataloader=dataloaders['val'], type='val')
                val_acc = val_result['val_acc']

                val_acc_str = 'val_acc: {:6.4f}.'.format(val_acc)
                print(val_acc_str)

                if args.log_file:
                    with open(os.path.join(args.log_dir, args.log_file), 'a+') as log_file:
                        log_file.write('stage 1 ' + 'epoch ' + str(epoch) + ', ' + val_acc_str + '\n')

        if args.use_cuda:
            torch.cuda.synchronize()

        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, 'weight_stage' + str(1) + '_epoch' + str(epoch) + '.pth')
            torch.save(model.state_dict(), save_path)

        if args.use_cuda:
            torch.cuda.empty_cache()

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

            leaves_out, final_out, final_features, penultimate_out = model(image)

            target = torch.ones([leaves_out[0].shape[0]])

            if args.use_cuda:
                target = target.cuda()

            loss = loss_func((leaves_out, final_out, final_features, penultimate_out), label, target)

            if args.use_cuda:
                loss = loss.cuda()

            loss.backward()

            if args.use_cuda:
                torch.cuda.synchronize()

            optimizers[1].step()

            if args.use_cuda:
                torch.cuda.synchronize()

            cur_time = datetime.datetime.now()

            loss_records.append(loss.detach().item())

            print('train_step: {:-8d} / {:d}, loss: {:6.4f}'
                  .format(cur_step, train_epoch_step, loss.detach().item()), flush=True)

            print(cur_time - last_time)

        if learning_rate_schedulers[1]:
            learning_rate_schedulers[1].step()

        print('stage 2')
        print('epoch: {:-4d}, start_epoch: {:-4d}, epoch_num: {:-4d}.'
              .format(epoch, args.start_epoch2, args.epoch_num2))

        if args.type == 'val':
            if (epoch + 1) % args.val_interval == 0:
                val_result = test(args, model=model, dataloader=dataloaders['val'], type='val')
                val_acc = val_result['val_acc']

                val_acc_str = 'val_acc: {:6.4f}.'.format(val_acc)
                print(val_acc_str)

                if args.log_file:
                    with open(os.path.join(args.log_dir, args.log_file), 'a+') as log_file:
                        log_file.write('stage 2 ' + 'epoch ' + str(epoch) + ', ' + val_acc_str + '\n')

        if args.use_cuda:
            torch.cuda.synchronize()

        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, 'weight_stage' + str(2) + '_epoch' + str(epoch) + '.pth')
            torch.save(model.state_dict(), save_path)

        if args.use_cuda:
            torch.cuda.empty_cache()
