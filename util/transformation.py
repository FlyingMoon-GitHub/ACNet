# -*- coding: utf-8 -*-

import copy
import random
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image


class MyTransformation(object):
    def __init__(self, transformation_list):
        self.transformation_list = transformation_list

    def __call__(self, imgs):
        if isinstance(imgs, Image.Image):
            imgs = [imgs]

        assert len(imgs) >= 1
        size = None
        for img_no, img in enumerate(imgs):
            assert isinstance(img, Image.Image)
        #     if img_no == 0:
        #         size = img.size
        #     else:
        #         assert size == img.size

        trans_imgs = copy.deepcopy(imgs)

        for trans in self.transformation_list:
            if isinstance(trans, transforms.RandomRotation):
                angle = trans.get_params(trans.degrees)
                for img_no, trans_img in enumerate(trans_imgs):
                    trans_imgs[img_no] = F.rotate(trans_img, angle, trans.resample, trans.expand, trans.center)
            elif isinstance(trans, transforms.RandomCrop):
                i, j, th, tw = trans.get_params(trans_imgs[0], trans.size)
                for img_no, trans_img in enumerate(trans_imgs):
                    trans_imgs[img_no] = F.crop(trans_img, i, j, th, tw)
            elif isinstance(trans, transforms.RandomResizedCrop):
                i, j, h, w = trans.get_params(trans_imgs[0], trans.scale, trans.ratio)
                for img_no, trans_img in enumerate(trans_imgs):
                    trans_imgs[img_no] = F.resized_crop(trans_img, i, j, h, w, trans.size, trans.interpolation)
            elif isinstance(trans, transforms.ColorJitter):
                color_transform = trans.get_params(trans.brightness, trans.contrast,
                                                   trans.saturation, trans.hue)
                for img_no, trans_img in enumerate(trans_imgs):
                    trans_imgs[img_no] = color_transform(trans_img)
            elif isinstance(trans, transforms.RandomHorizontalFlip):
                random_value = random.random()
                if random_value < trans.p:
                    for img_no, trans_img in enumerate(trans_imgs):
                        trans_imgs[img_no] = F.hflip(trans_img)
            elif isinstance(trans, transforms.RandomVerticalFlip):
                random_value = random.random()
                if random_value < trans.p:
                    for img_no, trans_img in enumerate(trans_imgs):
                        trans_imgs[img_no] = F.vflip(trans_img)
            elif isinstance(trans,
                            (transforms.Resize, transforms.CenterCrop, transforms.ToTensor, transforms.Normalize)):
                for img_no, trans_img in enumerate(trans_imgs):
                    trans_imgs[img_no] = trans(trans_img)
            else:
                print(trans)
                raise NotImplementedError()

        if len(imgs) == 1:
            trans_imgs = trans_imgs[0]

        return trans_imgs


def getTransformation(args, type):
    assert type in ['train', 'val', 'test']

    transformation_list = []

    transformation_list.append(transforms.Resize(size=(int(args.target_size / 0.75), int(args.target_size / 0.75))))

    if type == 'train':
        transformation_list.append(transforms.RandomCrop(size=(args.target_size, args.target_size)))
        transformation_list.append(transforms.RandomRotation(degrees=20))
        transformation_list.append(transforms.ColorJitter(brightness=0.1))
        transformation_list.append(transforms.RandomHorizontalFlip(p=0.5))
        # transformation_list.append(transforms.RandomVerticalFlip(p=0.5))
    elif type in ['val', 'test']:
        transformation_list.append(transforms.CenterCrop(size=(args.target_size, args.target_size)))

    transformation_list.append(transforms.ToTensor()),
    transformation_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    transformation = MyTransformation(transformation_list)

    return transformation
