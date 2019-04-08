# -*- coding:utf-8 -*-
import numpy as np
import os
from pyheatmap.heatmap import HeatMap


def save_heatmap(density_map, blob, imgs_dir, output_dir, down_sample=True):
    """
    生成热力图并保存
    :param density_map: 2d-array, 密度图
    :param blob: dict
    :param imgs_dir: 图片目录
    :param output_dir: 结果保存目录
    :param down_sample: bool, 是否有下采样
    :return:
    """
    img = blob['data']  # 图片数组, shape(h, w, 1)
    img_name = blob['fname']  # 图片文件名
    print('generating heatmap for', img_name)

    # 如果密度图进行下采样4倍, 则需要还原到原始大小
    if down_sample:
        den_resized = np.zeros((density_map.shape[0] * 4, density_map.shape[1] * 4))
        for i in range(den_resized.shape[0]):
            for j in range(den_resized.shape[1]):
                den_resized[i][j] = density_map[int(i / 4)][int(j / 4)] / 16
        density_map = den_resized

    h, w = img.shape[:2]
    density_map = density_map * 1000
    data = []
    for row in range(h):
        for col in range(w):
            for k in range(int(density_map[row][col])):
                data.append([col + 1, row + 1])
    # 生成heatmap
    hm = HeatMap(data, base=os.path.join(imgs_dir, img_name))
    # 保存
    hm.heatmap(save_as=os.path.join(output_dir, 'heatmap_'+img_name.split('.')[0]+'.png'))
