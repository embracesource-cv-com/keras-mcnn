# -*- coding:utf-8 -*-
import os
import cv2
import csv
import argparse
from scipy.io import loadmat
from data_preparation.density_map import gen_density_map
from config import current_config as cfg


def main(args):
    dataset = args.dataset
    cfg.init_path(dataset)  # 初始化路径名
    image_path = os.path.join(cfg.ORIGIN_DATA_PATH, 'test_data/images')
    gt_path = os.path.join(cfg.ORIGIN_DATA_PATH, 'test_data/ground_truth')
    gt_path_csv = os.path.join(cfg.ORIGIN_DATA_PATH, 'test_data/ground_truth_csv')

    if not os.path.exists(gt_path_csv):
        os.makedirs(gt_path_csv)
    if dataset == 'A':
        num_images = 182
    else:
        num_images = 316

    for i in range(1, num_images + 1):
        if i % 10 == 0:
            print('Processing {}/{} files'.format(i, num_images),
                  '\nwriting to {}'.format(''.join([gt_path_csv, 'IMG_', str(i), '.csv'])))
        image_info = loadmat(os.path.join(gt_path, 'GT_IMG_{}.mat'.format(i)))['image_info']
        input_img_path = os.path.join(image_path, 'IMG_{}.jpg'.format(i))
        im = cv2.imread(input_img_path, 0)
        points = image_info[0][0][0][0][0] - 1
        im_density = gen_density_map(im, points)
        with open(os.path.join(gt_path_csv, 'IMG_{}.csv'.format(i)), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(im_density)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="the dataset you want to create", choices=['A', 'B'])
    args = parser.parse_args()
    main(args)
