# This is based on direct_clustering.py
# This script performs superpixel selection using the overlaps with the estimated road mask.
# This is basically the implementation of "Distantly Supervised Road Segmentation" method in
# https://arxiv.org/abs/1708.06118

import sys  # NOQA isort:skip
sys.path.insert(0, 'models')  # NOQA isort:skip
sys.path.insert(0, 'datasets')  # NOQA isort:skip
sys.path.insert(0, '.')  # NOQA isort:skip

import argparse
import glob
import json
import os
import random
import time

from PIL import Image
import chainer
from chainer import cuda
from chainer import datasets
from chainer import serializers
from chainer.dataset import concat_examples
import chainer.functions as F
from chainercv import evaluations
import cupy as cp
import cv2 as cv
import drn
import matplotlib.pyplot as plt
import numpy as np
from resize_image_dataset import ResizeImageDataset
from scipy.ndimage import measurements
from skimage.segmentation import felzenszwalb
from skimage.segmentation import slic
from zipped_cityscapes_road_dataset import ZippedCityscapesRoadDataset

chainer.config.train = False
random.seed(1111)
np.random.seed(1111)
cp.random.seed(1111)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
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
    parser.add_argument('--felzenszwalb_scale', type=float, default=500.0)
    parser.add_argument('--felzenszwalb_sigma', type=float, default=0.9)
    parser.add_argument('--felzenszwalb_min_size', type=int, default=20)
    parser.add_argument('--overlap_threshold', type=float, default=0.01)
    parser.add_argument('--n_slic_segments', type=int, default=100)
    parser.add_argument(
        '--use_feature_maps', type=int, nargs='*', default=[7])
    parser.add_argument('--out_dir', type=str, default='data/test_images')
    parser.add_argument(
        '--img_file_list', type=str, default=None,
        help='data/random300_images.txt')
    parser.add_argument(
        '--label_file_list', type=str, default=None,
        help='data/random300_label.txt')
    parser.add_argument(
        '--cityscapes_img_dir', type=str, default=None,
        help='data/cityscapes/leftImg8bit/train')
    parser.add_argument(
        '--cityscapes_label_dir', type=str, default=None,
        help='data/cityscapes/gtFine/train')
    parser.add_argument(
        '--cityscapes_img_zip', type=str, default=None,
        help='If it\'s given, ZippedCityscapesRoadDataset will be use.'
             'e.g., data/cityscapes_random_300_train_imgs.0.zip')
    parser.add_argument(
        '--cityscapes_label_zip', type=str, default=None,
        help='If it\'s given, ZippedCityscapesRoadDataset will be use.'
             'e.g., data/cityscapes_random_300_train_labels.0.zip')
    parser.add_argument('--camera_param_dir', type=str, default='data/camera')
    parser.add_argument('--start_index', type=int)
    parser.add_argument('--end_index', type=int)
    args = parser.parse_args()
    args.resize_shape = tuple(args.resize_shape)
    if not os.path.exists(args.out_dir):
        try:
            os.makedirs(args.out_dir)
        except Exception:
            pass
    return args


def weighted_average(a, b, axis=0):
    return (a * b[:, None]).sum(axis=axis) / b.sum(axis=axis)


def kmeans(k, X, weights=None, n_iter=1000):
    assert X.ndim == 2, 'The ndim of X should be 2'
    if weights is not None:
        assert weights.ndim == 1, 'The ndim of weights should be 2'
        assert len(weights) == len(X), 'The lengths of X and weights should be same'

    xp = cuda.get_array_module(X)
    weights_other = 1 - weights

    # Initial assignment
    assign = xp.zeros((X.shape[0],))

    # Prior, put pixels with high weight in the first cluster
    prior_weight_threshold = float(xp.sort(weights)[len(weights) // 2])  # The center weight
    assign[weights > prior_weight_threshold] = 0
    cond = weights <= prior_weight_threshold  # Binary map
    idx = xp.arange(int(cond.sum())) % (k - 1) + 1
    xp.random.shuffle(idx)
    assign[cond] = idx  # Randomly assign initial clusters
    centers = xp.stack([X[assign == i].mean(axis=0) for i in xp.arange(k)])

    for _ in range(n_iter):
        # calculate distances and label
        distances = xp.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_assign = xp.argmin(distances, axis=1).astype(np.int32)
        if xp.all(new_assign == assign):
            break
        assign = new_assign

        # calc new centers
        mask = assign == 0
        masked_X = X[mask]
        masked_w = weights[mask]
        centers[0] = weighted_average(masked_X, masked_w, axis=0)
        for j in range(1, k):
            mask = assign == j
            masked_X = X[mask]
            masked_w = weights_other[mask]
            centers[j] = weighted_average(masked_X, masked_w, axis=0)

        done = False
        for j in range(k):
            if (assign == j).sum() == 0:
                print(('Terminate KMeans iteration due to {}-th cluster is '
                       'empty').format(j))
                done = True
                break
        if done:
            break

    return cuda.to_cpu(assign)


def create_label_mask(label):
    # From the official script distributed here:
    # https://github.com/mcordts/cityscapesScripts/
    # We mark 'void' categories as -1
    # 'road' class as 1
    # otherwise 0
    assert label.ndim == 2

    ids_label = np.zeros_like(label, dtype=np.int32)
    void_class_idss = [0, 1, 2, 3, 4, 5, 6]
    for i in void_class_idss:
        ids_label[label == i] = -1

    road_ids = [7]
    for i in road_ids:
        ids_label[label == i] = 1

    return ids_label


def create_prior(h, w, y_rel_pos=0.75, x_rel_pos=0.5, y_rel_sigma=0.1, x_rel_sigma=0.2):
    xcoord, ycoord = np.meshgrid(np.arange(w), np.arange(h))

    ymean, xmean = int(h * y_rel_pos), int(w * x_rel_pos)
    y_sigma = h * y_rel_sigma
    x_sigma = w * x_rel_sigma

    weights = np.exp(
        -((ycoord - ymean) ** 2 / (2 * y_sigma) ** 2
          + (xcoord - xmean) ** 2 / (2 * x_sigma) ** 2))

    assert weights.shape == (h, w), 'The shape of weights should be (h, w)'

    return weights


def batch_weighted_kmeans(args, feature_maps, weights):
    clustering_results = kmeans(k=args.n_clusters, X=feature_maps, weights=weights)
    if args.gpu >= 0:
        clustering_results = cuda.to_cpu(clustering_results)
    return clustering_results


def save_image(args, img, road_mask, label, clustering_result, img_fn):
    plt.clf()
    fig, axes = plt.subplots(2, 2)
    fig.set_dpi(300)
    axes[0, 0].axis('off')
    axes[0, 1].axis('off')
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')

    # Show result
    axes[0, 0].imshow(img / 255.)
    axes[0, 0].imshow(road_mask, alpha=0.4, cmap=plt.cm.Set1_r)
    axes[0, 0].set_title('Estimated road mask (input image overlayed)',
                         fontsize=8)
    # Show labels
    axes[0, 1].imshow(label == 1)
    axes[0, 1].set_title('Ground truth road mask', fontsize=8)
    # Show clustering result
    axes[1, 0].imshow(clustering_result)
    axes[1, 0].set_title('All clusters', fontsize=8)
    # Show road estimation
    axes[1, 1].imshow(clustering_result == 0)
    axes[1, 1].set_title('Estimated road mask', fontsize=8)

    plt.savefig(os.path.join(args.out_dir, os.path.basename(img_fn)),
                bbox_inches='tight')


def save_info(
        img_fn, label_fn, road_mask, clustering_result, label, elapsed_times,
        st_all):
    out_fn = os.path.splitext(os.path.basename(img_fn))[0]
    np.save(os.path.join(args.out_dir, out_fn), road_mask.astype(np.uint8))
    out_fn = out_fn + '_all_cluster'
    np.save(os.path.join(args.out_dir, out_fn),
            clustering_result.astype(np.uint8))

    ret = evaluations.calc_semantic_segmentation_confusion([road_mask], [label])
    try:
        TP = int(ret[1, 1])
        FP = int(ret[0, 1])
        FN = int(ret[1, 0])
    except Exception as e:
        # print(str(type(e)), e)
        # print('ret:', ret)
        # print('road_mask:', road_mask.shape, np.unique(road_mask))
        # print('label:', label.shape, np.unique(label))
        TP, FP, FN = 0, 0, 0
    precision = float(TP / (TP + FP)) if TP + FP > 0 else None
    recall = float(TP / (TP + FN)) if TP + FN > 0 else None
    if TP == 0 and FP == 0 and FN == 0:
        iou = [0, 0]
    else:
        iou = evaluations.calc_semantic_segmentation_iou(ret)

    with open(os.path.join(args.out_dir, 'result.json'), 'a') as fp:
        result_info = {
            'img_fn': img_fn,
            'label_fn': label_fn,
            'road_iou': iou[1],
            'non_road_iou': iou[0],
            'precision': precision,
            'recall': recall,
            'TP': TP,
            'FP': FP,
            'FN': FN
        }
        result_info.update(vars(args))
        elapsed_times['elapsed_time'] = time.time() - st_all
        result_info.update(elapsed_times)
        print(json.dumps(result_info), file=fp)

    return result_info


def batch_superpixel(args, imgs):
    superpixels = []
    if args.superpixel_method == 'felzenszwalb':
        for img in imgs:
            superpixels.append(felzenszwalb(
                img.transpose(1, 2, 0) / 255.,
                scale=args.felzenszwalb_scale,
                sigma=args.felzenszwalb_sigma,
                min_size=args.felzenszwalb_min_size))
    elif args.superpixel_method == 'slic':
        for img in imgs:
            superpixels.append(
                slic(img.transpose(1, 2, 0), args.n_slic_segments))
    superpixels = np.asarray(superpixels)
    return superpixels


def estimate_road_mask(imgs, img_fns, labels, label_fns, model, args):
    st_all = time.time()
    elapsed_times = {}
    xp = model.xp

    imgs = model.xp.asarray(imgs)
    st = time.time()
    _, maps = model.batch_predict(imgs)
    elapsed_times['time_feature_maps'] = time.time() - st
    use_maps = [maps[i] for i in args.use_feature_maps]

    # Calculate superpixels
    st = time.time()
    orig_imgs = np.asarray([np.asarray(Image.open(fn), dtype=np.uint8).transpose(2, 0, 1) for fn in img_fns])
    superpixels = batch_superpixel(args, orig_imgs)  # (n, h, w)
    elapsed_times['time_superpixel'] = time.time() - st

    # Concat feature maps
    feature_maps = F.concat(use_maps, axis=1).array
    n, c, h, w = feature_maps.shape
    xycoord = np.stack(np.meshgrid(np.arange(w), np.arange(h))).reshape(2, -1)[None, ...].repeat(n, axis=0)
    xycoord = xp.asarray(xycoord.transpose(0, 2, 1).reshape(-1, 2), dtype=np.int32)  # (N, H * W, 2) -> (N * H * W, 2)
    feature_maps = feature_maps.transpose(0, 2, 3, 1).reshape(n * h * w, c)
    feature_maps = xp.concatenate([feature_maps, xycoord], axis=1)

    # Create road prior
    st = time.time()
    prior = create_prior(h, w, args.y_rel_pos, args.x_rel_pos, args.y_rel_sigma, args.x_rel_sigma)
    prior = prior.reshape(1, h * w).repeat(n, axis=0).reshape(n * h * w)
    prior = xp.asarray(prior)
    elapsed_times['time_prior'] = time.time() - st

    # Weighted KMeans to obtain estimated road mask
    st = time.time()
    clustering_results = batch_weighted_kmeans(args, feature_maps, prior)
    elapsed_times['time_kmeans'] = time.time() - st
    clustering_results = clustering_results.reshape(n, h, w)
    road_masks = clustering_results == 0
    if args.gpu >= 0:
        road_masks = cuda.to_cpu(road_masks)

    for img_fn, label_fn, clustering_result, road_mask, superpixel in zip(
            img_fns, label_fns, clustering_results, road_masks, superpixels):
        # Load image
        img = np.asarray(Image.open(img_fn), dtype=np.uint8)

        # Load labels
        label = np.asarray(Image.open(label_fn), dtype=np.uint8)
        label = create_label_mask(label.copy())

        # Merge the overlapped superpixels using road_mask
        if road_mask.shape != superpixel.shape:
            h, w = superpixel.shape
            road_mask = cv.resize(road_mask.astype(np.uint8), (w, h), interpolation=cv.INTER_NEAREST)
        refined_roadmap = np.zeros_like(road_mask)
        n_pred_road_pixels = float(np.sum(road_mask))
        for idx in np.unique(superpixel):
            sp_mask = superpixel == idx
            overlap = float(np.sum(np.asarray(sp_mask, dtype=np.int32) * road_mask))
            if n_pred_road_pixels > 0 and (overlap / float(n_pred_road_pixels)) > args.overlap_threshold:
                refined_roadmap[sp_mask] = 1

        if clustering_result.shape != label.shape:
            h, w = label.shape
            clustering_result = cv.resize(clustering_result.astype(np.uint8), (w, h), interpolation=cv.INTER_NEAREST)

        save_image(args, img, refined_roadmap, label, clustering_result, img_fn)
        result_info = save_info(img_fn, label_fn, refined_roadmap, clustering_result, label, elapsed_times, st_all)

        print('Road IoU:', result_info['road_iou'], os.path.basename(img_fn))


def create_dataset(args):
    if args.cityscapes_img_zip is not None \
            and args.cityscapes_label_zip is not None:
        dataset = ZippedCityscapesRoadDataset(
            args.cityscapes_img_zip, args.cityscapes_label_zip,
            args.resize_shape, standardize=False)
    elif args.img_file_list is not None \
            and args.label_file_list is not None:
        il = [l.strip() for l in open(args.img_file_list).readlines() if l]
        ll = [l.strip() for l in open(args.label_file_list).readlines() if l]
        img_d = ResizeImageDataset(il, args.resize_shape, dtype=np.float32)
        label_d = ResizeImageDataset(ll, None, dtype=np.uint8)
        dataset = datasets.TupleDataset(img_d, label_d)
        dataset.img_fns = img_d._paths
        dataset.label_fns = label_d._paths
    else:
        val_img_files = {
            '_'.join(os.path.basename(fn).split('_')[:3]): fn
            for fn in glob.glob(
                os.path.join(args.cityscapes_img_dir, '*', '*.png'))}
        val_label_files = {
            '_'.join(os.path.basename(fn).split('_')[:3]): fn
            for fn in glob.glob(
                os.path.join(args.cityscapes_label_dir, '*', '*labelIds.png'))}
        img_fns = []
        label_fns = []
        for key in val_label_files.keys():
            img_fns.append(val_img_files[key])
            label_fns.append(val_label_files[key])
        img_d = ResizeImageDataset(
            img_fns, args.resize_shape, dtype=np.float32)
        label_d = ResizeImageDataset(label_fns, None, dtype=np.uint8)
        dataset = datasets.TupleDataset(img_d, label_d)
        dataset.img_fns = img_d._paths
        dataset.label_fns = label_d._paths
    return dataset


def create_model(args):
    model = drn.drn_c_26(out_map=True, out_middle=True)
    serializers.load_npz('models/drn_c_26.npz', model)
    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)
    return model


if __name__ == '__main__':
    args = get_args()
    dataset = create_dataset(args)
    model = create_model(args)

    for i in range(args.start_index, args.end_index, args.batchsize):
        if i + args.batchsize >= args.end_index:
            # To keep the batchsize
            i = args.end_index - args.batchsize
            end_i = args.end_index
        else:
            end_i = i + args.batchsize
        imgs, labels = concat_examples(dataset[i:end_i])
        img_fns = dataset.img_fns[i:end_i]
        label_fns = dataset.label_fns[i:end_i]
        estimate_road_mask(imgs, img_fns, labels, label_fns, model, args)
