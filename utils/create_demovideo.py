#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys  # NOQA isort:skip
sys.path.insert(0, 'models')  # NOQA isort:skip

import argparse
import glob
import os

from PIL import Image
import chainer
from chainer import serializers
from chainercv import transforms
from chainercv import utils
import numpy as np
from tqdm import tqdm
import cv2 as cv

from segnet_basic import SegNetBasic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--demoVideo_dir', type=str,
                        default='data/cityscapes/leftImg8bit/demoVideo')
    args = parser.parse_args()
    return args


def main():
    mean = np.array([
        7.315835921071366954e+01,
        8.290891754262415247e+01,
        7.239239876194160672e+01], dtype=np.float32)
    std = np.array([
        4.161211675686322309e+01,
        4.221582767516605372e+01,
        4.048309952494058450e+01], dtype=np.float32)
    resize_shape = (512, 1024)

    chainer.config.train = False

    args = get_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    model = SegNetBasic(n_class=2, pred_shape=(1024, 2048))
    serializers.load_npz(
        args.snapshot, model, path='updater/model:main/predictor/')
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    for fn in tqdm(
            sorted(glob.glob(os.path.join(args.demoVideo_dir, '*', '*.png')))):
        img = utils.read_image(fn)
        img = transforms.resize(img, resize_shape, 3)
        img -= mean[:, None, None]
        img /= std[:, None, None]
        label = model.predict([img])[0]
        cv.imwrite(os.path.join(args.out_dir, os.path.basename(fn)), label)


if __name__ == '__main__':
    main()
