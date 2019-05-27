# -*- coding:utf-8 -*-
import os
import csv
import cv2
import math
import random
from scipy.io import loadmat
import argparse
from data_preparation.density_map import gen_density_map
from data_preparation.sample import random_crop
from utils.file_utils import recreate_dir
from config import current_config as cfg


def main(args):
    seed = 123456
    random.seed(seed)
    dataset = args.dataset
    if dataset == 'A':
        num_images = 300
    else:
        num_images = 400

    cfg.init_path(dataset)  # 初始化路径名
    image_path = os.path.join(cfg.ORIGIN_DATA_PATH, 'train_data/images')
    gt_path = os.path.join(cfg.ORIGIN_DATA_PATH, 'train_data/ground_truth')

    num_val = math.ceil(num_images * 0.1)  # 验证集数量（数据集的10%）
    indices = list(range(1, num_images + 1))  # 编号从1开始
    random.shuffle(indices)

    # 重建目录
    recreate_dir(cfg.TRAIN_PATH)
    recreate_dir(cfg.TRAIN_GT_PATH)
    recreate_dir(cfg.VAL_PATH)
    recreate_dir(cfg.VAL_GT_PATH)

    # 逐个图像采样
    for idx in range(num_images):
        i = indices[idx]
        if (idx + 1) % 10 == 0:
            print('Processing {}/{} files'.format(idx + 1, num_images))
        # 加载图片
        input_img_name = os.path.join(image_path, 'IMG_{}.jpg'.format(i))
        im = cv2.imread(input_img_name, 0)
        # 加载对应标注
        image_info = loadmat(os.path.join(gt_path, 'GT_IMG_{}.mat'.format(i)))['image_info']
        points = image_info[0][0][0][0][0] - 1
        # 生成密度图
        im_density = gen_density_map(im, points)
        # 随机采样9张子图
        image_samples, density_samples = random_crop(im, im_density, 9)

        for j, (image, density) in enumerate(zip(image_samples, density_samples)):
            # 保存
            image_prefix = "{}_{}".format(i, j)  # 图像编号_裁剪编号
            dir_im, dir_den = (cfg.VAL_PATH, cfg.VAL_GT_PATH) if (idx + 1) < num_val else (
                cfg.TRAIN_PATH, cfg.TRAIN_GT_PATH)
            path_im = os.path.join(dir_im, "{}.jpg".format(image_prefix))
            path_den = os.path.join(dir_den, "{}.csv".format(image_prefix))
            cv2.imwrite(path_im, image)
            with open(path_den, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(density)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="the dataset you want to create", choices=['A', 'B'])
    args = parser.parse_args()
    main(args)
