# -*- coding:utf-8 -*-
import os
import csv
import cv2
import math
import random
from scipy.io import loadmat
import gen_density_map
import argparse


def main(args):
    seed = 123456
    random.seed(seed)
    dataset = args.dataset
    N = 9  # 在每张原始图片上采用9张小图片
    dataset_name = ''.join(['shanghaitech_part_', dataset, '_patches_', str(N)])
    path = ''.join(['../data/original/shanghaitech/part_', dataset, '_final/train_data/images/'])
    output_path = '../data/formatted_trainval_{}/'.format(dataset)
    train_path_img = ''.join((output_path, dataset_name, '/train/'))
    train_path_den = ''.join((output_path, dataset_name, '/train_den/'))
    val_path_img = ''.join((output_path, dataset_name, '/val/'))
    val_path_den = ''.join((output_path, dataset_name, '/val_den/'))
    gt_path = ''.join(['../data/original/shanghaitech/part_', dataset, '_final/train_data/ground_truth/'])

    for i in [output_path, train_path_img, train_path_den, val_path_img, val_path_den]:
        if not os.path.exists(i):
            os.makedirs(i)

    if dataset == 'A':
        num_images = 300
    else:
        num_images = 400

    num_val = math.ceil(num_images * 0.1)  # 验证集数量（数据集的10%）
    indices = list(range(1, num_images + 1))
    random.shuffle(indices)

    for idx in range(num_images):
        i = indices[idx]
        if (idx+1) % 10 == 0:
            print('Processing {}/{} files'.format(idx+1, num_images))
        # 加载图片
        input_img_name = ''.join((path, 'IMG_', str(i), '.jpg'))
        im = cv2.imread(input_img_name, 0)
        # 加载对应标注
        image_info = loadmat(''.join((gt_path, 'GT_IMG_', str(i), '.mat')))['image_info']
        annPoints = image_info[0][0][0][0][0] - 1
        # 生成密度图
        im_density = gen_density_map.gen_density_map(im, annPoints)

        h, w = im.shape
        wn2, hn2 = w / 8, h / 8  # 小图片大小为原始图片的1/4，这两个变量为小图片宽和高的1/2，所以除以8
        wn2, hn2 = int(wn2 / 8) * 8, int(hn2 / 8) * 8  # 确保为8的整数倍
        # 得到（小图片）中心点的采样范围
        xmin, xmax = wn2, w - wn2
        ymin, ymax = hn2, h - hn2
        # 在原图中随机采样9张小图片
        for j in range(1, N + 1):
            # 随机采样得到中心点坐标x,y
            x = math.floor((xmax - xmin) * random.random() + xmin)
            y = math.floor((ymax - ymin) * random.random() + ymin)
            # 得到左上角以及右下角坐标
            x1, y1 = x - wn2, y - hn2
            x2, y2 = x + wn2, y + hn2
            # 在原图以及密度图中crop出对应区域
            im_sampled = im[y1:y2, x1:x2]
            im_density_sampled = im_density[y1:y2, x1:x2]

            # 保存
            img_idx = ''.join((str(i), '_', str(j)))
            path_img, path_den = (val_path_img, val_path_den) if (idx+1) < num_val else (train_path_img, train_path_den)
            cv2.imwrite(''.join([path_img, img_idx, '.jpg']), im_sampled)
            with open(''.join([path_den, img_idx, '.csv']), 'w', newline='') as fout:
                writer = csv.writer(fout)
                writer.writerows(im_density_sampled)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="the dataset you want to create", choices=['A', 'B'])
    args = parser.parse_args()
    main(args)
