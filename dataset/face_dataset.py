'''
Author: supermantx
Date: 2024-07-16 09:27:50
LastEditTime: 2025-05-13 08:17:59
Description: 
'''

import os
import numbers

import mxnet as mx
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
import cv2 as cv
from matplotlib import pyplot as plt


class MXFaceDataset(Dataset):

    def __init__(self, root_dir, transform=None, flag=True):
        super(MXFaceDataset, self).__init__()
        if not transform:
            # masked decoder期间不用发杂的数据增强
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5,
                                                                0.5]),
            ])
        else:
            self.transform = transform
        self.flag = flag
        self.imgrec = mx.recordio.MXIndexedRecordIO(
            os.path.join(root_dir, 'train.idx'),
            os.path.join(root_dir, 'train.rec'), 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
        self.colors = [(0, 0, 255), (0, 255, 255), (255, 0, 255), (0, 255, 0),
                       (255, 0, 0)]

    def __get_raw_item(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        img_raw = mx.image.imdecode(img).asnumpy()
        print(header)
        return img_raw, header.label

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        # 水平旋转
        if not self.flag:
            flip_flag = False
            if torch.rand(1) < 0.5:
                flip_flag = True
                sample = F.hflip(sample)
            return sample, label
        else:
            return sample, label

    def show(self, index):
        img_raw, header = self.__get_raw_item(index)
        anno = list(map(int, header))
        cv.rectangle(img_raw, (anno[0], anno[1]), (anno[2], anno[3]),
                     (0, 255, 0), 1)
        for i in range(5):
            cv.circle(img_raw, (anno[2 * i + 4], anno[2 * i + 5]), 1,
                      self.colors[i], 1)
        plt.imshow(img_raw)
        # plt.show()

    def __len__(self):
        # return 100000
        return len(self.imgidx)
