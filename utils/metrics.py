# -*- coding:utf-8 -*-
import keras.backend as K


def mae(y_true, y_pred):
    return K.abs(K.sum(y_true) - K.sum(y_pred))


def mse(y_true, y_pred):
    return (K.sum(y_true) - K.sum(y_pred)) * (K.sum(y_true) - K.sum(y_pred))
