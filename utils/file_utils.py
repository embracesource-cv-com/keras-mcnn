# -*- coding: utf-8 -*-
"""
   File Name：     file_utils.py
   Description :  
   Author :       mick.yi
   Date：          2019/5/27
"""
import os
import shutil


def get_sub_files(dir_path, recursive=False):
    """
    获取目录下所有文件名
    :param dir_path:
    :param recursive: 是否递归
    :return:
    """
    file_paths = []
    for dir_name in os.listdir(dir_path):
        cur_dir_path = os.path.join(dir_path, dir_name)
        if os.path.isdir(cur_dir_path) and recursive:
            file_paths = file_paths + get_sub_files(cur_dir_path)
        else:
            file_paths.append(cur_dir_path)
    return file_paths


def delete_dir(dir_path, recursive=False):
    """
    删除目录
    :param dir_path:
    :param recursive:
    :return:
    """
    if os.path.exists(dir_path):
        if recursive:
            fps = get_sub_files(dir_path)
            for sub_dir in fps:
                if os.path.isdir(sub_dir):
                    delete_dir(sub_dir, recursive)
        shutil.rmtree(dir_path)


def recreate_dir(dir_path, recursive=False):
    """
    重建目录
    :param dir_path:
    :param recursive:
    :return:
    """
    delete_dir(dir_path, recursive)
    os.mkdir(dir_path)


def make_dir(dir_path):
    """
    创建目录
    :param dir_path:
    :return:
    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
