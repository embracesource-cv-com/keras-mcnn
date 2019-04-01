# -*- coding:utf-8 _*-
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from model import MCNN
from utils.data_loader import DataLoader
from utils.metrics import mae, mse
import config as cfg
import os
import argparse


def main(args):
    dataset = args.dataset  # 'A' or 'B'

    train_path = cfg.TRAIN_PATH.format(dataset)
    train_gt_path = cfg.TRAIN_GT_PATH.format(dataset)
    val_path = cfg.VAL_PATH.format(dataset)
    val_gt_path = cfg.VAL_GT_PATH.format(dataset)
    # 生成训练数据和验证数据
    train_data_gen = DataLoader(train_path, train_gt_path, shuffle=True, gt_downsample=True)
    val_data_gen = DataLoader(val_path, val_gt_path, shuffle=False, gt_downsample=True)
    x_train, y_train = train_data_gen.get_all()
    x_val, y_val = val_data_gen.get_all()

    # 定义模型
    input_shape = (None, None, 1)
    model = MCNN(input_shape)
    adam = Adam(lr=1e-4)
    model.compile(loss='mse', optimizer=adam, metrics=[mae, mse])

    # 定义callback
    checkpointer_best_train = ModelCheckpoint(
        filepath=os.path.join(cfg.MODEL_DIR, 'mcnn_'+dataset+'_train_generator.hdf5'),
        monitor='loss', verbose=1, save_best_only=True, mode='min'
    )
    callback_list = [checkpointer_best_train]

    # 训练
    print('Training Part_{} ...'.format(dataset))
    model.fit(
        x=x_train, y=y_train, batch_size=1, epochs=cfg.EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callback_list
    )

    # model.fit_generator(train_data_gen.flow(cfg.TRAIN_BATCH_SIZE),
    #                     steps_per_epoch=train_data_gen.num_samples // cfg.TRAIN_BATCH_SIZE,
    #                     validation_data=val_data_gen.flow(cfg.VAL_BATCH_SIZE),
    #                     validation_steps=val_data_gen.num_samples // cfg.VAL_BATCH_SIZE,
    #                     epochs=cfg.EPOCHS,
    #                     callbacks=callback_list,
    #                     verbose=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="the dataset you want to train", choices=['A', 'B'])
    args = parser.parse_args()
    main(args)
