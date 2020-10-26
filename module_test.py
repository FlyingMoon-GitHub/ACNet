import argparse
import cv2
import numpy as np
import os
import torch
from torch.autograd import Variable

from data.dataset import *
from model.ufnet import *
from util.config import *

def modelPropagationTest():
    parser = argparse.ArgumentParser()
    parser.add_argument('--class_num', dest='class_num',
                        default=4, type=int)
    parser.add_argument('--target_size', dest='target_size',
                        default=512, type=int)
    parser.add_argument('--lambda', dest='lambda_',
                        default=0.75, type=float)

    args = parser.parse_args()

    image_root_path = r'C:\Users\FlyingMoon\PycharmProjects\FNet\data\cervical\image'
    image_dir = ['2', '3', '4', '5']
    image_reagent = ['VIA3', 'VILI']
    image_num = len(image_dir)

    x_aux = np.zeros((image_num, 3, args.target_size, args.target_size))
    x_prim = np.zeros((image_num, 3, args.target_size, args.target_size))

    for i in range(image_num):
        temp = cv2.imread(os.path.join(image_root_path, image_dir[i], image_reagent[0] + '.jpg'))
        temp = cv2.resize(temp, (args.target_size, args.target_size))
        # temp = np.array(temp)
        temp = np.transpose(temp, (2, 0, 1))
        x_aux[i, :] = temp

        temp = cv2.imread(os.path.join(image_root_path, image_dir[i], image_reagent[1] + '.jpg'))
        temp = cv2.resize(temp, (args.target_size, args.target_size))
        # temp = np.array(temp)
        temp = np.transpose(temp, (2, 0, 1))
        x_prim[i, :] = temp

    x_aux = x_aux.astype(np.float)
    x_aux = torch.from_numpy(x_aux)
    if torch.cuda.is_available():
        x_aux = x_aux.cuda()
    x_aux = Variable(x_aux, requires_grad=True).float()

    x_prim = x_prim.astype(np.float)
    x_prim = torch.from_numpy(x_prim)
    if torch.cuda.is_available():
        x_prim = x_prim.cuda()
    x_prim = Variable(x_prim, requires_grad=True).float()

    model = UFNet(getModelConfig(args, 'train'))
    model.summary()

    y_aux, y_prim = model(x_aux, x_prim)

    print(y_aux, y_prim)

    target = torch.Tensor([1, 0, 3, 2]).long()
    print(target)

    loss_aux = nn.CrossEntropyLoss()(y_aux, target)
    loss_prim = nn.CrossEntropyLoss()(y_prim, target)

    print(loss_aux, loss_prim)

    lambda_ = 0.75
    loss = loss_aux * (1 - lambda_) + loss_prim * lambda_

    print(loss)

    loss.backward()

if __name__ == '__main__':
    modelPropagationTest()
