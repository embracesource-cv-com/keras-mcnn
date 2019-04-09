# -*- coding:utf-8 -*-
from utils.data_loader import DataLoader
from utils.heatmap import save_heatmap
import numpy as np
import config as cfg
import argparse
import os


def main(args):
    dataset = args.dataset  # 'A' or 'B'
    output_dir = os.path.join(cfg.HM_GT_PATH, 'Part_{}'.format(dataset))

    for _dir in [cfg.HM_GT_PATH, output_dir]:
        if not os.path.exists(_dir):
            os.mkdir(_dir)

    test_path = cfg.TEST_PATH.format(dataset)
    test_gt_path = cfg.TEST_GT_PATH.format(dataset)
    # load data
    data_loader = DataLoader(test_path, test_gt_path, shuffle=False, gt_downsample=True)

    # create heatmaps
    print('Creating heatmaps for Part_{} ...'.format(dataset))
    for blob in data_loader:
        gt = blob['gt']
        # create and save heatmap
        gt = np.squeeze(gt)  # shape(1, h, w, 1) -> shape(h, w)
        save_heatmap(gt, blob, test_path, output_dir)
    print('All Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="the dataset you want to create heatmaps for", choices=['A', 'B'])
    args = parser.parse_args()
    main(args)
