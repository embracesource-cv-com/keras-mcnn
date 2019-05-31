# -*- coding: utf-8 -*-
"""
   File Name：     sample.py
   Description :   采样
   Author :       mick.yi
   Date：          2019/5/27
"""
import numpy as np


def random_crop(image, density_map, crop_num=9):
    """
    随机裁剪图像和密度图像
    :param image: 灰度图，[H,W]
    :param density_map: 图像对应的密度图 [H,W]
    :param crop_num: 裁剪数
    :return:
    """
    height, width = image.shape
    crop_h, crop_w = height / 4, width / 4  # 裁剪尺寸为原始图像的1/4
    crop_h, crop_w = int(crop_h / 16) * 16, int(crop_w / 16) * 16  # 可以被16整除
    # 确定起始点的范围   0<=start;  end=start+crop_size<image_size;    0<=start<image_size-crop_size
    image_crops = []
    density_crops = []
    for _ in range(crop_num):
        start_h = np.random.randint(0, height - crop_h)
        start_w = np.random.randint(0, width - crop_w)
        image_crops.append(image[start_h:start_h + crop_h, start_w:start_w + crop_w])
        density_crops.append(density_map[start_h:start_h + crop_h, start_w:start_w + crop_w])

    return image_crops, density_crops


def main():
    image = np.random.randn(100, 100)
    density = np.random.randn(100, 100)

    crops_im, crops_den = random_crop(image, density, 9)

    print(crops_im[0].shape)
    print(crops_den[1].shape)


if __name__ == '__main__':
    main()
