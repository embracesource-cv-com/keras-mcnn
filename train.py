# -*- coding:utf-8 -*-
"""
   File Name:     train.py
   Description:   training entrance
   Author:        steven.yi
   date:          2019/04/17
"""
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from model import MCNN
from utils.data_loader import DataLoader
from utils.metrics import mae, mse
from config import current_config as cfg
import os
import argparse


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    dataset = args.dataset  # 'A' or 'B'
    cfg.init_path(dataset)  # 初始化路径名

    # 加载数据生成器
    train_data_gen = DataLoader(cfg.TRAIN_PATH,
                                cfg.TEST_GT_PATH,
                                batch_size=cfg.TRAIN_BATCH_SIZE,
                                shuffle=True,
                                gt_downsample=True)
    val_data_gen = DataLoader(cfg.VAL_PATH,
                              cfg.VAL_GT_PATH,
                              batch_size=cfg.VAL_BATCH_SIZE,
                              shuffle=False,
                              gt_downsample=True)

    # 定义模型
    input_shape = (None, None, 1)
    model = MCNN(input_shape)
    adam = Adam(lr=1e-4)
    model.compile(loss='mse', optimizer=adam, metrics=[mae, mse])

    # 定义callback
    checkpointer_best_train = ModelCheckpoint(
        filepath=cfg.WEIGHT_PATH,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='min'
    )
    callback_list = [checkpointer_best_train]

    # 训练
    print('Training Part_{} ...'.format(dataset))
    model.fit_generator(train_data_gen,
                        validation_data=val_data_gen,
                        epochs=cfg.EPOCHS,
                        callbacks=callback_list,
                        verbose=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="the dataset you want to train", choices=['A', 'B'])
    args = parser.parse_args()
    main(args)
