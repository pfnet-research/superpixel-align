#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys  # NOQA isort:skip
sys.path.insert(0, '.')  # NOQA isort:skip
sys.path.insert(0, 'datasets')  # NOQA isort:skip

import os
import argparse
import time

from PIL import Image
from chainer.dataset import concat_examples
import chainer.functions as F
import numpy as np

from batch_spalign_kmeans import batch_create_prior
from batch_spalign_kmeans import batch_superpixel
from batch_spalign_kmeans import batch_superpixel_align
from batch_spalign_kmeans import batch_weighted_kmeans
from batch_spalign_kmeans import create_model
import cv2 as cv
from resize_image_dataset import ResizeImageDataset


def estimate_road_mask(imgs, img_fns, model, args):
    st_all = time.time()
    elapsed_times = {}

    _, maps = model.batch_predict(model.xp.asarray(imgs))
    use_maps = [maps[i] for i in args.use_feature_maps]

    # Concat feature maps
    feature_maps = F.concat(use_maps, axis=1)

    # Cals superpixels
    st = time.time()
    superpixels = batch_superpixel(args, imgs)
    elapsed_times['time_superpixel'] = time.time() - st

    # Superpixel Align
    st = time.time()
    superpixel_features, n_superpixels_per_image = batch_superpixel_align(
        args, model, imgs, superpixels, feature_maps)
    elapsed_times['time_roialign'] = time.time() - st

    # Create road prior
    st = time.time()
    superpixel_weights = batch_create_prior(args, superpixels)
    elapsed_times['time_prior'] = time.time() - st

    # Weighted KMeans to obtain estimated road mask
    st = time.time()
    clustering_results, road_masks = batch_weighted_kmeans(
        args, superpixels, superpixel_features,
        superpixel_weights, n_superpixels_per_image)
    elapsed_times['time_kmeans'] = time.time() - st

    for img_fn, road_mask in zip(img_fns, road_masks):

        # Load image
        img = np.asarray(Image.open(img_fn), dtype=np.uint8)

        if road_mask.shape != tuple(args.label_shape):
            h, w = args.label_shape
            road_mask = cv.resize(
                road_mask.astype(np.uint8), (w, h),
                interpolation=cv.INTER_NEAREST)

        save_fn = os.path.join(args.out_dir, os.path.basename(img_fn))
        cv.imwrite(save_fn, road_mask)
        print(save_fn)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--img_list_fn', type=str, default='data/demoVideo_fns.txt')
    parser.add_argument(
        '--label_shape', type=int, nargs=2, default=[1024, 2048])
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--start_index', type=int)
    parser.add_argument('--end_index', type=int)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument(
        '--superpixel_method', type=str, default='felzenszwalb',
        choices=['felzenszwalb', 'slic'])
    parser.add_argument('--n_clusters', type=int, default=4)
    parser.add_argument('--y_rel_pos', type=float, default=0.75)
    parser.add_argument('--x_rel_pos', type=float, default=0.5)
    parser.add_argument('--y_rel_sigma', type=float, default=0.1)
    parser.add_argument('--x_rel_sigma', type=float, default=0.1)
    parser.add_argument(
        '--n_anchors', type=int, default=10,
        help='Number of points sampled from inside of superpixel for RoIAlign')
    parser.add_argument(
        '--n_neighbors', type=int, default=4,
        help='Number of neighboring pixels on feature map to calculate'
             'bilinear interpolation of feature for an anchor')
    parser.add_argument(
        '--without_pos', action='store_true', default=False,
        help='If True, the coordinates of center of mass of a superpixel will '
             'not be appended to the superpixel align feature vector')
    parser.add_argument(
        '--horizontal_line_filtering', action='store_true', default=False,
        help='If True, it filters out all estimated label pixels above the '
             'horizonal line calculated using camera parameters into 0')
    parser.add_argument(
        '--resize_shape', type=int, nargs=2, default=[224, 224],
        help='The resize shape for the input image. It shouldbe '
             '(HEIGHT, WIDTH) order')
    parser.add_argument(
        '--batchsize', type=int, default=30,
        help='The size for batch clustering')
    parser.add_argument('--felzenszwalb_scale', type=float, default=300.0)
    parser.add_argument('--felzenszwalb_sigma', type=float, default=0.8)
    parser.add_argument('--felzenszwalb_min_size', type=int, default=20)
    parser.add_argument('--n_slic_segments', type=int, default=100)
    parser.add_argument(
        '--use_feature_maps', type=int, nargs='*', default=[7])
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    model = create_model(args)
    img_fns = list(sorted([fn.strip() for fn in open(args.img_list_fn)]))
    print('img_fns:', len(img_fns))
    img_d = ResizeImageDataset(
        img_fns, args.resize_shape, dtype=np.float32)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for i in range(args.start_index, args.end_index, args.batchsize):
        if i + args.batchsize >= args.end_index:
            # To keep the batchsize
            i = args.end_index - args.batchsize
            end_i = args.end_index
        else:
            end_i = i + args.batchsize
        imgs = concat_examples(img_d[i:end_i])
        img_fns = img_d._paths[i:end_i]
        estimate_road_mask(imgs, img_fns, model, args)
