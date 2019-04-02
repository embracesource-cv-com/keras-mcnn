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
    kernel_size = 15  # 高斯核size
    sigma = 4.0  # 标准差

    for point in anno_points:
        # 人头的中心点坐标
        x, y = min(w-1, abs(math.floor(point[0]))), min(h-1, abs(math.floor(point[1])))
        # 左上角坐标以及右下角坐标
        x1, y1 = x - kernel_size // 2, y - kernel_size // 2
        x2, y2 = x + kernel_size // 2 + 1, y + kernel_size // 2 + 1

        out_of_bounds = False
        dx1, dy1, dx2, dy2 = 0, 0, 0, 0  # 越界的偏移量
        # 以下四个if用来判断两个顶角的x,y是否越界
        if x1 < 0:
            dx1 = abs(x1)
            x1 = 0
            out_of_bounds = True
        if y1 < 0:
            dy1 = abs(y1)
            y1 = 0
            out_of_bounds = True
        if x2 > w:
            dx2 = x2 - w
            x2 = w
            out_of_bounds = True
        if y2 > h:
            dy2 = y2 - h
            y2 = h
            out_of_bounds = True

        if out_of_bounds:
            # 如果越界，则调整高斯核的大小
            kernel_h = kernel_size - dy1 - dy2
            kernel_w = kernel_size - dx1 - dx2
            # 生成大小为(kernel_h, kernel_w)的高斯核
            H = np.multiply(cv2.getGaussianKernel(kernel_h, sigma), (cv2.getGaussianKernel(kernel_w, sigma)).T)
        else:
            # 生成大小为(15, 15)的高斯核
            H = np.multiply(cv2.getGaussianKernel(kernel_size, sigma), (cv2.getGaussianKernel(kernel_size, sigma)).T)

        density_map[y1:y2, x1:x2] += H
    return density_map
