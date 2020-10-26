# -*- coding: utf-8 -*-

import argparse

import cv2
import os
from PIL import Image, ImageFile
from torch.utils.data import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

class MyDataset(Dataset):
    def __init__(self, config):
        super(MyDataset, self).__init__()

        self.root_path = config['root_path']
        self.dataset = config['dataset']
        self.img_dir = config['img_dir']
        self.anno_dir = config['anno_dir']
        self.anno_file = config['anno_file']
        self.class_num = config['class_num']

        self.anno = []
        with open(os.path.join(self.root_path, self.dataset, self.anno_dir, self.anno_file), 'r') as f:
            line = f.readline().replace('\n', '')
            while line:
                line_items = line.split(' ')
                self.anno.append((line_items[0], int(line_items[1])))
                line = f.readline().replace('\n', '')

        assert self.anno, 'No available data?'

        self.img_aug = config['img_aug']

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root_path, self.dataset, self.img_dir, self.anno[index][0])).convert('RGB')
        image = self.img_aug(image)

        return image, self.anno[index][1]


if __name__ == '__main__':
    from util.arg_parse import *
    from util.config import *
    args = getArgs()

    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    args.root_path = '..\\data'

    train_data_config = getDatasetConfig(args, 'train')
    val_data_config = getDatasetConfig(args, 'val')

    train_dataset = MyDataset(train_data_config)
    val_dataset = MyDataset(val_data_config)

    import random
    idx1 = random.randint(0, len(train_dataset))
    item1 = train_dataset[idx1]

    idx2 = random.randint(0, len(val_dataset))
    item2 = val_dataset[idx2]

    import matplotlib.pyplot as plt

    print(item1[1], item2[1])
    plt.imshow(item1[0].permute(1, 2, 0))
    plt.show()

    plt.imshow(item2[0].permute(1, 2, 0))
    plt.show()
