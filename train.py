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
import tensorflow as tf
import keras


def set_gpu_growth():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto(allow_soft_placement=True)  # because no supported kernel for GPU devices is available
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    keras.backend.set_session(session)


def main(args):
    set_gpu_growth()
    dataset = args.dataset  # 'A' or 'B'
    cfg.init_path(dataset)  # 初始化路径名

    # 加载数据生成器
    train_data_gen = DataLoader(cfg.TRAIN_PATH,
                                cfg.TRAIN_GT_PATH,
                                batch_size=cfg.TRAIN_BATCH_SIZE,
                                shuffle=True,
                                gt_downsample=True,
                                mean=cfg.MEAN,
                                std=cfg.STD)
    val_data_gen = DataLoader(cfg.VAL_PATH,
                              cfg.VAL_GT_PATH,
                              batch_size=cfg.VAL_BATCH_SIZE,
                              shuffle=False,
                              gt_downsample=True,
                              mean=cfg.MEAN,
                              std=cfg.STD)

    # 定义模型
    input_shape = (None, None, 1)
    model = MCNN(input_shape)
    adam = Adam(lr=1e-4)
    model.compile(loss='mse', optimizer=adam, metrics=[mae, mse])
    # 加载与训练模型
    if args.weight_path is not None:
        model.load_weights(args.weight_path, by_name=True)

    # 定义callback
    checkpoint = ModelCheckpoint(
        filepath=cfg.WEIGHT_PATH,
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=True,
        mode='min',
        period=5
    )
    callback_list = [checkpoint]

    # 训练
    print('Training Part_{} ...'.format(dataset))
    model.fit_generator(train_data_gen,
                        validation_data=val_data_gen,
                        epochs=cfg.EPOCHS,
                        initial_epoch=args.init_epoch,
                        callbacks=callback_list,
                        use_multiprocessing=True,
                        workers=4,
                        verbose=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="the dataset you want to train", choices=['A', 'B'])
    parser.add_argument("--init_epoch", type=int, default=0, help="init epoch")
    parser.add_argument("--weight_path", type=str, default=None, help="weight path")
    args = parser.parse_args()
    main(args)
