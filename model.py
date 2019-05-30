# -*- coding:utf-8 -*-
"""
   File Name:     model.py
   Description:   model definition
   Author:        steven.yi
   date:          2019/04/17
"""
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Concatenate


def MCNN(input_shape=None):
    inputs = Input(shape=input_shape)

    # column 1
    column_1 = Conv2D(16, (9, 9), padding='same', activation='relu', name='col1_conv1')(inputs)
    column_1 = MaxPooling2D(2)(column_1)
    column_1 = Conv2D(32, (7, 7), padding='same', activation='relu', name='col1_conv2')(column_1)
    column_1 = MaxPooling2D(2)(column_1)
    column_1 = Conv2D(16, (7, 7), padding='same', activation='relu', name='col1_conv3')(column_1)
    column_1 = Conv2D(8, (7, 7), padding='same', activation='relu', name='col1_conv4')(column_1)

    # column 2
    column_2 = Conv2D(20, (7, 7), padding='same', activation='relu', name='col2_conv1')(inputs)
    column_2 = MaxPooling2D(2)(column_2)
    column_2 = Conv2D(40, (5, 5), padding='same', activation='relu', name='col2_conv2')(column_2)
    column_2 = MaxPooling2D(2)(column_2)
    column_2 = Conv2D(20, (5, 5), padding='same', activation='relu', name='col2_conv3')(column_2)
    column_2 = Conv2D(10, (5, 5), padding='same', activation='relu', name='col2_conv4')(column_2)

    # column 3
    column_3 = Conv2D(24, (5, 5), padding='same', activation='relu', name='col3_conv1')(inputs)
    column_3 = MaxPooling2D(2)(column_3)
    column_3 = Conv2D(48, (3, 3), padding='same', activation='relu', name='col3_conv2')(column_3)
    column_3 = MaxPooling2D(2)(column_3)
    column_3 = Conv2D(24, (3, 3), padding='same', activation='relu', name='col3_conv3')(column_3)
    column_3 = Conv2D(12, (3, 3), padding='same', activation='relu', name='col3_conv4')(column_3)

    # merge feature map of 3 columns in last dimension
    merges = Concatenate(axis=-1)([column_1, column_2, column_3])
    # density map
    density_map = Conv2D(1, (1, 1), padding='same', activation='relu', name='density_conv')(merges)

    model = Model(inputs=inputs, outputs=density_map)
    return model
