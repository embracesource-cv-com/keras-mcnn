# -*- coding:utf-8 _*-
from keras.models import load_model
from utils.data_loader import DataLoader
import numpy as np
import config as cfg
import argparse
import os
import cv2


def save_density_map(save_dir, density_map, fname='results.png'):
    """
    保存预测密度图
    :param save_dir: 保存目录
    :param density_map: 密度图矩阵
    :param fname: 保存文件名
    :return:
    """
    density_map = 255*density_map/np.max(density_map)
    density_map = density_map[0][0]
    cv2.imwrite(os.path.join(save_dir, fname), density_map)


def main(args):
    dataset = args.dataset  # 'A' or 'B'
    if dataset == 'A':
        model_path = './trained_models/mcnn_A_train.hdf5'
    else:
        model_path = './trained_models/mcnn_B_train_generator.hdf5'

    output_dir = './output_{}/'.format(dataset)
    density_maps_dir = os.path.join(output_dir, 'density_maps')  # directory to save predicted density map
    results_txt = os.path.join(output_dir, 'results.txt')  # file to save predicted results
    for _dir in [output_dir, density_maps_dir]:
        if not os.path.exists(_dir):
            os.mkdir(_dir)

    test_path = cfg.TEST_PATH.format(dataset)
    test_gt_path = cfg.TEST_GT_PATH.format(dataset)
    # load test set
    data_loader = DataLoader(test_path, test_gt_path, shuffle=False, gt_downsample=True)
    # load model
    model = load_model(model_path)

    # test
    print('Testing Part_{} ...'.format(dataset))
    mae = 0.0
    mse = 0.0
    for blob in data_loader:
        img = blob['data']
        gt = blob['gt']
        pred = model.predict(np.expand_dims(img, axis=0))
        gt_count = np.sum(gt)
        pred_count = np.sum(pred)
        mae += abs(gt_count - pred_count)
        mse += ((gt_count - pred_count) * (gt_count - pred_count))
        # # save density map
        # save_density_map(density_maps_dir, pred, blob['fname'].split('.')[0] + '.png')

        # save results
        with open(results_txt, 'a') as f:
            line = '<{}> {:.2f} -- {:.2f}\n'.format(blob['fname'].split('.')[0], gt_count, pred_count)
            f.write(line)

    mae = mae / data_loader.num_samples
    mse = np.sqrt(mse / data_loader.num_samples)
    print('MAE: %0.2f, MSE: %0.2f' % (mae, mse))
    with open(results_txt, 'a') as f:
        f.write('MAE: %0.2f, MSE: %0.2f' % (mae, mse))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="the dataset you want to predict", choices=['A', 'B'])
    args = parser.parse_args()
    main(args)
