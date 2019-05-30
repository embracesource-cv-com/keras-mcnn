# -*- coding:utf-8 -*-
"""
   File Name:     config.py
   Description:   global config parameters
   Author:        steven.yi
   date:          2019/04/17
"""


class Config(object):
    ORIGIN_DATA_PATH = "/opt/dataset/crowd_counting/shanghaitech/original/part_{}_final/"

    # 训练集目录
    TRAIN_PATH = '/opt/dataset/crowd_counting/shanghaitech/formatted_trainval_{0}/train'
    # 训练集Ground-Truth目录
    TRAIN_GT_PATH = '/opt/dataset/crowd_counting/shanghaitech/formatted_trainval_{0}/train_den'

    # 验证集目录
    VAL_PATH = '/opt/dataset/crowd_counting/shanghaitech/formatted_trainval_{0}/val'
    # 验证集Ground_Truth目录
    VAL_GT_PATH = '/opt/dataset/crowd_counting/shanghaitech/formatted_trainval_{0}/val_den'

    # 测试集目录
    TEST_PATH = '/opt/dataset/crowd_counting/shanghaitech/original/part_{}_final//test_data/images/'
    # 测试集Ground_Truth目录
    TEST_GT_PATH = '/opt/dataset/crowd_counting/shanghaitech/original/part_{}_final/test_data/ground_truth_csv/'

    EPOCHS = 50
    TRAIN_BATCH_SIZE = 1
    VAL_BATCH_SIZE = 1

    def init_path(self, ds):
        """
        初始化路径
        :param ds: 数据集名称 A or B
        :return: None
        """
        self.WEIGHT_PATH = '/tmp/mcnn-' + ds + '.{epoch:03d}.h5'  # 权重存放目录
        self.ORIGIN_DATA_PATH = self.ORIGIN_DATA_PATH.format(ds)
        self.TRAIN_PATH = self.TRAIN_PATH.format(ds)
        self.TRAIN_GT_PATH = self.TRAIN_GT_PATH.format(ds)

        self.VAL_PATH = self.VAL_PATH.format(ds)
        self.VAL_GT_PATH = self.VAL_GT_PATH.format(ds)

        self.TEST_PATH = self.TEST_PATH.format(ds)
        self.TEST_GT_PATH = self.TEST_GT_PATH.format(ds)


current_config = Config()

if __name__ == '__main__':
    current_config.init_path('A')
    print(current_config.ORIGIN_DATA_PATH)
