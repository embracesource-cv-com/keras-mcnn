import os
import csv
import cv2
import math
import random
from scipy.io import loadmat
from . import gen_density_map
import argparse


def main(args):
    seed = 95461354
    random.seed(seed)
    N = 9
    dataset = args.dataset
    dataset_name = ''.join(['shanghaitech_part_', dataset, '_patches_', str(N)])
    path = ''.join(['./data/original/shanghaitech/part_', dataset, '_final/train_data/images/'])
    output_path = './data/formatted_trainval_{}/'.format(dataset)
    train_path_img = ''.join((output_path, dataset_name, '/train/'))
    train_path_den = ''.join((output_path, dataset_name, '/train_den/'))
    val_path_img = ''.join((output_path, dataset_name, '/val/'))
    val_path_den = ''.join((output_path, dataset_name, '/val_den/'))
    gt_path = ''.join(['./data/original/shanghaitech/part_', dataset, '_final/train_data/ground_truth/'])

    for i in [output_path, train_path_img, train_path_den, val_path_img, val_path_den]:
        if not os.path.exists(i):
            os.makedirs(i)

    if dataset == 'A':
        num_images = 300
    else:
        num_images = 400

    num_val = math.ceil(num_images*0.1)
    indices = list(range(1, num_images+1))
    random.shuffle(indices)

    for idx in range(1, num_images+1):
        i = indices[idx-1]
        if idx % 10 == 0:
            print('Processing {}/{} files'.format(idx, num_images))
        image_info = loadmat(''.join((gt_path, 'GT_IMG_', str(i), '.mat')))['image_info']
        input_img_name = ''.join((path, 'IMG_', str(i), '.jpg'))
        im = cv2.imread(input_img_name, 0)
        h, w = im.shape
        wn2, hn2 = w / 8, h / 8
        wn2, hn2 = int(wn2 / 8) * 8, int(hn2 / 8) * 8
        annPoints = image_info[0][0][0][0][0] - 1
        if w <= wn2 * 2:
            im = cv2.resize(im, [h, wn2*2+1], interpolation=cv2.INTER_LANCZOS4)
            annPoints[:, 0] = annPoints[:, 0] * 2 * wn2 / w
        if h <= hn2 * 2:
            im = cv2.resize(im, [hn2*2+1, w], interpolation=cv2.INTER_LANCZOS4)
            annPoints[:, 1] = annPoints[:, 1] * 2 * hn2 / h
        h, w = im.shape
        a_w, b_w = wn2 + 1, w - wn2
        a_h, b_h = hn2 + 1, h - hn2

        im_density = gen_density_map.gen_density_map(im, annPoints)
        for j in range(1, N+1):
            x = math.floor((b_w - a_w) * random.random() + a_w)
            y = math.floor((b_h - a_h) * random.random() + a_h)
            x1, y1 = x - wn2, y - hn2
            x2, y2 = x + wn2 - 1, y + hn2 - 1
            im_sampled = im[y1-1:y2, x1-1:x2]
            im_density_sampled = im_density[y1-1:y2, x1-1:x2]
            # annPoints_sampled = annPoints[
            #     list(
            #         set(np.where(np.squeeze(annPoints[:,0]) > x1)[0].tolist()) &
            #         set(np.where(np.squeeze(annPoints[:,0]) < x2)[0].tolist()) &
            #         set(np.where(np.squeeze(annPoints[:,1]) > y1)[0].tolist()) &
            #         set(np.where(np.squeeze(annPoints[:,1]) < y2)[0].tolist())
            #     )
            # ]
            #
            # annPoints_sampled[:, 0] = annPoints_sampled[:, 0] - x1
            # annPoints_sampled[:, 1] = annPoints_sampled[:, 1] - y1
            img_idx = ''.join((str(i), '_',str(j)))

            # if idx < num_val:
            #     cv2.imwrite(''.join([val_path_img, img_idx, '.jpg']), im_sampled)
            #     with open(''.join([val_path_den, img_idx, '.csv']), 'w', newline='') as fout:
            #         writer = csv.writer(fout)
            #         writer.writerows(im_density_sampled)
            # else:
            #     cv2.imwrite(''.join([train_path_img, img_idx, '.jpg']), im_sampled)
            #     with open(''.join([train_path_den, img_idx, '.csv']), 'w', newline='') as fout:
            #         writer = csv.writer(fout)
            #         writer.writerows(im_density_sampled)

            path_img, path_den = (val_path_img, val_path_den) if idx+1 < num_val else (train_path_img, train_path_den)
            cv2.imwrite(''.join([path_img, img_idx, '.jpg']), im_sampled)
            with open(''.join([path_den, img_idx, '.csv']), 'w', newline='') as fout:
                writer = csv.writer(fout)
                writer.writerows(im_density_sampled)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="the dataset you want to create", choices=['A', 'B'])
    args = parser.parse_args()
    main(args)
