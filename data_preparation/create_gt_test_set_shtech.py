import os
import cv2
import csv
import argparse
from scipy.io import loadmat
import gen_density_map


def main(args):
    dataset = args.dataset
    path = ''.join(['./data/original/shanghaitech/part_', dataset, '_final/test_data/images/'])
    gt_path = ''.join(['./data/original/shanghaitech/part_', dataset, '_final/test_data/ground_truth/'])
    gt_path_csv = ''.join(['./data/original/shanghaitech/part_', dataset, '_final/test_data/ground_truth_csv/'])
    if not os.path.exists(gt_path_csv):
        os.makedirs(gt_path_csv)
    if dataset == 'A':
        num_images = 182
    else:
        num_images = 316

    for i in range(1, num_images+1):
        if i % 10 == 0:
            print('Processing {}/{} files'.format(i, num_images), '\nwriting to {}'.format(''.join([gt_path_csv, 'IMG_', str(i), '.csv'])))
        image_info = loadmat(''.join((gt_path, 'GT_IMG_', str(i), '.mat')))['image_info']
        input_img_name = ''.join((path, 'IMG_', str(i), '.jpg'))
        im = cv2.imread(input_img_name, 0)
        annPoints = image_info[0][0][0][0][0] - 1
        im_density = gen_density_map.gen_density_map(im, annPoints)
        with open(''.join([gt_path_csv, 'IMG_', str(i), '.csv']), 'w', newline='') as fout:
            writer = csv.writer(fout)
            writer.writerows(im_density)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="the dataset you want to create", choices=['A', 'B'])
    args = parser.parse_args()
    main(args)
