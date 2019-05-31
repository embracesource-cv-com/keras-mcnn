# -*- coding:utf-8 -*-
"""
   File Name:     test.py
   Description:   testing entrance
   Author:        steven.yi
   date:          2019/04/17
"""
from utils.data_loader import DataLoader
from utils.heatmap import save_heatmap
from model import MCNN
import numpy as np
from config import current_config as cfg
import argparse
import os


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset = args.dataset  # 'A' or 'B'
    output_dir = args.output_dir
    weight_path = args.weight_path
    cfg.init_path(dataset)

    heatmaps_dir = os.path.join(output_dir, 'heatmaps')  # directory to save heatmap
    results_txt = os.path.join(output_dir, 'results.txt')  # file to save predicted results
    for _dir in [output_dir, heatmaps_dir]:
        if not os.path.exists(_dir):
            os.mkdir(_dir)

    # load test set
    data_loader = DataLoader(cfg.TEST_PATH,
                             cfg.TEST_GT_PATH,
                             shuffle=False,
                             gt_downsample=True)
    # load model
    model = MCNN(input_shape=(None, None, 1))
    model.load_weights(weight_path, by_name=True)

    # test
    print('Testing Part_{} ...'.format(dataset))
    mae = 0.0
    mse = 0.0
    for idx, (img, gt) in enumerate(data_loader):
        filename = data_loader.filename_list[idx]
        pred = model.predict(img)
        pred *= cfg.STD
        pred += cfg.MEAN
        gt_count = np.sum(gt)
        pred_count = np.sum(pred)
        mae += abs(gt_count - pred_count)
        mse += ((gt_count - pred_count) * (gt_count - pred_count))
        # create and save heatmap
        pred = np.squeeze(pred)  # shape(1, h, w, 1) -> shape(h, w)
        # save_heatmap(pred, img, filename, heatmaps_dir)
        # save results
        with open(results_txt, 'a') as f:
            line = '<{}> {:.2f} -- {:.2f}\n'.format(filename, gt_count, pred_count)
            f.write(line)

    mae = mae / len(data_loader)
    mse = np.sqrt(mse / len(data_loader))
    print('MAE: %0.2f, MSE: %0.2f' % (mae, mse))
    with open(results_txt, 'a') as f:
        f.write('MAE: %0.2f, MSE: %0.2f' % (mae, mse))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="A", help="the dataset you want to predict", choices=['A', 'B'])
    parser.add_argument("--weight_path", type=str, default="tmp/mcnn-A.h5", help="weight path")
    parser.add_argument("--output_dir", default="./", help="output directory to save predict heatmaps")

    args = parser.parse_args()
    main(args)
