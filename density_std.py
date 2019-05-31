# -*- coding: utf-8 -*-
"""
   File Name：     density_std.py
   Description :   计算密度图的标准差
   Author :       mick.yi
   Date：          2019/5/30
"""
import argparse
import numpy as np
from utils.data_loader import DataLoader
from config import current_config as cfg


def main(args):
    dataset = args.dataset  # 'A' or 'B'
    cfg.init_path(dataset)  # 初始化路径名
    # 加载数据生成器
    train_data_gen = DataLoader(cfg.TRAIN_PATH,
                                cfg.TRAIN_GT_PATH,
                                batch_size=cfg.TRAIN_BATCH_SIZE,
                                shuffle=True,
                                gt_downsample=True)
    dens = [np.ravel(den) for im, den in train_data_gen]
    dens = np.concatenate(dens, axis=0)
    print("mean:{},std:{}".format(np.mean(dens), np.std(dens)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="the dataset you want to train", choices=['A', 'B'])
    args = parser.parse_args()
    main(args)
