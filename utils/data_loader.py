# -*- coding:utf-8 -*-
"""
   File Name:     data_loader.py
   Description:   data preprocess and load
   Author:        steven.yi
   date:          2019/04/17
"""
import numpy as np
import cv2
import os
import pandas as pd
from keras.utils import Sequence


class DataLoader(Sequence):
    def __init__(self, data_path, gt_path, batch_size=1, shuffle=False, gt_downsample=False, mean=0., std=1.):
        """
        :param data_path: 图片文件路径
        :param gt_path: ground truth路径
        :param batch_size:
        :param shuffle: bool, 是否打乱数据
        :param gt_downsample: bool, 是否下采样
        """
        self.data_path = data_path
        self.gt_path = gt_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.gt_downsample = gt_downsample
        self.mean = mean
        self.std = std
        self.filename_list = [filename for filename in os.listdir(data_path)]

    def __getitem__(self, item):
        """

        :param item: 索引编号 int
        :return:
        """
        ix = np.arange(self.batch_size * item, self.batch_size * (item + 1))
        images, dens = [], []
        for index in ix:
            im, den = self._load(self.filename_list[index])
            images.append(im)
            dens.append(den)

        # 转为numpy返回
        return np.array(images), np.array(dens)

    def __len__(self):
        return len(self.filename_list) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.filename_list)

    def _load(self, filename):
        """

        :param filename: 图像文件名  "abc.jpg"
        :return: image: numpy数组[H,W,1]
        :return: density_map: numpy数组 [h,w,1]
        """
        image_path = os.path.join(self.data_path, filename)
        density_name = os.path.splitext(filename)[0] + '.csv'
        density_path = os.path.join(self.gt_path, density_name)
        # 加载图像
        im = cv2.imread(image_path, 0)
        im = im.astype(np.float32, copy=False)
        # 保证长宽可以被4整除，网络的步长为4
        h, w = im.shape
        im = cv2.resize(im, (w // 4 * 4, h // 4 * 4))

        # 加载密度图
        den = pd.read_csv(density_path, header=None).values
        den = den.astype(np.float32, copy=False)
        if self.gt_downsample:
            den = cv2.resize(den, (w // 4, h // 4))
        scale_factor = w * h / ((w // 4) * (h // 4))
        den *= scale_factor
        # 减均值，除方差
        den -= self.mean
        den /= self.std

        # 扩展通道维返回
        return im[:, :, np.newaxis], den[:, :, np.newaxis]
