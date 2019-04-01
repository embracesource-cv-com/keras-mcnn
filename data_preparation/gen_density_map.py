# -*- coding:utf-8 _*-
import cv2
import math
import numpy as np


def gen_density_map(img, anno_points):
    """
    生成密度图矩阵
    :param img: 2d array, 图片数组
    :param anno_points: 2d array, 标注的人头坐标, shape(nums,2)
    :return: 2d array
    """
    density_map = np.zeros_like(img, dtype=np.float64)
    h, w = density_map.shape

    # 若没有标注数据
    if anno_points.size == 0:
        return density_map

    # 若只有一个标注数据
    if anno_points.shape[0] == 1:
        x = max(0, min(w-1, round(anno_points[0, 0])))
        y = max(0, min(h-1, round(anno_points[0, 1])))
        density_map[int(y), int(x)] = 255
        return density_map

    for point in anno_points:
        f_sz = 15
        sigma = 4.0
        H = np.multiply(cv2.getGaussianKernel(f_sz, sigma), (cv2.getGaussianKernel(f_sz, sigma)).T)
        x = min(w - 1, max(0, abs(math.floor(point[0]))))
        y = min(h - 1, max(0, abs(math.floor(point[1]))))
        if x >= w or y >= h:
            continue

        # 左上角坐标以及右下角坐标
        x1, y1 = x - f_sz // 2, y - f_sz // 2
        x2, y2 = x + f_sz // 2 + 1, y + f_sz // 2 + 1

        dfx1, dfy1, dfx2, dfy2 = 0, 0, 0, 0
        change = False
        if x1 < 0:
            dfx1 = abs(x1)
            x1 = 0
            change = True
        if y1 < 0:
            dfy1 = abs(y1)
            y1 = 0
            change = True
        if x2 > w:
            dfx2 = x2 - w
            x2 = w
            change = True
        if y2 > h:
            dfy2 = y2 - h
            y2 = h
            change = True
        x1h, y1h, x2h, y2h = 1 + dfx1, 1 + dfy1, f_sz - dfx2, f_sz - dfy2
        if change is True:
            H = np.multiply(cv2.getGaussianKernel(y2h-y1h+1, sigma), (cv2.getGaussianKernel(x2h-x1h+1, sigma)).T)

        density_map[y1:y2, x1:x2] += H
    return density_map
