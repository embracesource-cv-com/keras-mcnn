# -*- coding:utf-8 -*-
"""
   File Name:     metrics.py
   Description:   definition of metrics
   Author:        steven.yi
   date:          2019/04/17
"""
import keras.backend as K


def mae(y_true, y_pred):
    """
    mean absolute error
    :param y_true: shape(1, h, w, 1)
    :param y_pred: shape(1, h, w, 1)
    :return: float
    """
    shape = K.shape(y_pred)
    h, w = shape[1], shape[2]
    y_true = K.reshape(y_true, [-1, h * w])
    y_pred = K.reshape(y_pred, [-1, h * w])
    loss = K.abs(K.sum(y_true, axis=-1) - K.sum(y_pred, axis=-1))  # [n]
    return K.mean(loss)


def mse(y_true, y_pred):
    """
    mean square error
    :param y_true: shape(1, h, w, 1)
    :param y_pred: shape(1, h, w, 1)
    :return: float
    """
    shape = K.shape(y_pred)
    h, w = shape[1], shape[2]
    y_true = K.reshape(y_true, [-1, h * w])
    y_pred = K.reshape(y_pred, [-1, h * w])
    loss = K.square(K.sum(y_true, axis=-1) - K.sum(y_pred, axis=-1))
    return K.sqrt(K.mean(loss))


# def mae(y_true, y_pred):
#     return K.abs(K.sum(y_true) - K.sum(y_pred))
#
#
# def mse(y_true, y_pred):
#     return (K.sum(y_true) - K.sum(y_pred)) * (K.sum(y_true) - K.sum(y_pred))
