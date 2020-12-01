# -*- coding: utf-8 -*-

import datetime
import os
import torch
from torch import nn
from torch.autograd import Variable


def test(args, model, dataloader, type):
    assert type in ['val', 'test']

    result = {}

    loss_records = []

    gpu_ids = [int(i) for i in args.gpu_ids.split(',')]
    first_gpu_device = torch.device('cuda:' + str(gpu_ids[0]))

    batch_size = dataloader.batch_size
    epoch_step = len(dataloader)
    data_num = len(dataloader.dataset)

    correct = 0

    class_num = args.class_num
    confusion_matrix = [[0] * class_num for _ in range(class_num)]

    model.train(False)

    with torch.no_grad():

        cur_step = 0

        for batch_no, data in enumerate(dataloader):

            cur_step += 1

            image, label = data

            if args.use_cuda:
                image = Variable(image.to(first_gpu_device))
                label = Variable(label.to(first_gpu_device)).long()
            else:
                image = Variable(image)
                label = Variable(label).long()

            leaves_out, final_out = model(image)

            loss = nn.NLLLoss()(final_out, label)

            for out in leaves_out:
                loss = loss + nn.NLLLoss()(out, label)

            if args.use_cuda:
                loss = loss.to(first_gpu_device)

            loss_records.append(loss.detach().item())

            print(type + '_step: {:-8d} / {:d}, loss: {:6.4f}.'
                  .format(cur_step, epoch_step, loss.detach().item()), flush=True)

            _, pred = torch.topk(final_out, 1)
            correct += torch.sum((pred[:, 0] == label)).data.item()

            for i in range(label.shape[0]):
                confusion_matrix[label[i]][pred[i, 0]] += 1

    acc = correct / data_num

    result['loss_records'] = loss_records
    result[type + '_acc'] = acc
    result[type + '_cfs_mat'] = confusion_matrix

    return result
