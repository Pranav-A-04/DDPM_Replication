r"""
File to extract csv images from csv files for mnist dataset.
"""

import os
import cv2
import argparse
from tqdm import tqdm
import numpy as np
import _csv as csv

def extract_images(save_dir, csv_fname):
    assert os.path.exists(save_dir), "Directory {} to save images does not exist".format(save_dir)
    assert os.path.exists(csv_fname), "Csv file {} does not exist".format(csv_fname)
    with open(csv_fname) as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            im = np.zeros((784))
            im[:] = list(map(int, row[1:]))
            im = im.reshape((28,28))
            if not os.path.exists(os.path.join(save_dir, row[0])):
                os.mkdir(os.path.join(save_dir, row[0]))
            cv2.imwrite(os.path.join(save_dir, row[0], '{}.png'.format(idx)), im)
            if idx % 1000 == 0:
                print('Finished creating {} images in {}'.format(idx+1, save_dir))
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--mnist_train_csv_path', dest='mnist_train_csv_path',
                        default='/input/mnist-in-csv/mnist_train.csv', type=str)
    parser.add_argument('--mnist_test_csv_path', dest='mnist_test_csv_path',
                        default='/input/mnist-in-csv/mnist_test.csv', type=str)
    parser.add_argument('--mnist_train_images_path', dest='mnist_train_images_path',
                        default='/data/train/images', type=str)
    parser.add_argument('--mnist_test_images_path', dest='mnist_test_images_path',
                        default='/data/test/images', type=str)
    args = parser.parse_args()
    extract_images(args.mnist_train_images_path, args.mnist_train_csv_path)
    extract_images(args.mnist_test_images_path, args.mnist_test_csv_path)