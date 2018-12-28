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
    parser.add_argument('--felzenszwalb_scale', type=float, default=300.0)
    parser.add_argument('--felzenszwalb_sigma', type=float, default=0.8)
    parser.add_argument('--felzenszwalb_min_size', type=int, default=20)
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


def create_prior(superpixels, y_rel_pos=0.75, x_rel_pos=0.5, y_rel_sigma=0.1,
                 x_rel_sigma=0.2):
    h, w = superpixels.shape
    xcoord, ycoord = np.meshgrid(np.arange(w), np.arange(h))

    ymean, xmean = int(h * y_rel_pos), int(w * x_rel_pos)
    y_sigma = h * y_rel_sigma
    x_sigma = w * x_rel_sigma

    weights = np.exp(
        -((ycoord - ymean) ** 2 / (2 * y_sigma) ** 2
          + (xcoord - xmean) ** 2 / (2 * x_sigma) ** 2))

    superpixel_weights = np.array([])
    for idx in np.sort(np.unique(superpixels)):
        mean_weights = weights[superpixels == idx].mean()
        superpixel_weights = np.append(superpixel_weights, mean_weights)

    return superpixel_weights


def weighted_average(a, b, axis=0):
    return (a * b[:, None]).sum(axis=axis) / b.sum(axis=axis)


def kmeans(k, X, weights=None, n_iter=1000):
    xp = cuda.get_array_module(X)
    weights_other = 1 - weights

    # Initial assignment
    assign = xp.zeros((X.shape[0],))

    # Prior, put pixels with high weight in the first cluster
    prior_weight_threshold = float(xp.sort(weights)[len(weights) // 2])
    assign[weights > prior_weight_threshold] = 0
    cond = weights <= prior_weight_threshold
    idx = xp.arange(int(cond.sum())) % (k - 1) + 1
    xp.random.shuffle(idx)
    assign[cond] = idx
    centers = xp.stack(
        [X[assign == i].mean(axis=0) for i in xp.arange(k)])

    for _ in range(n_iter):
        # calculate distances and label
        distances = xp.linalg.norm(
            X[:, None, :] - centers[None, :, :], axis=2)
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


def weighted_kmeans(
        superpixels, superpixel_features, superpixel_weights, k,
        n_superpixels_per_image):
    result = kmeans(k=k, X=superpixel_features, weights=superpixel_weights)
    xp = cuda.get_array_module(superpixels)
    clustering_result = xp.zeros_like(superpixels)  # N, H, W

    i = 0
    for img_idx, n_superpixels in enumerate(n_superpixels_per_image):
        for superpixel_idx, cluster_id \
                in enumerate(result[i:i + n_superpixels]):
            clustering_result[img_idx][
                superpixels[img_idx] == superpixel_idx] = cluster_id
        i += n_superpixels

        if xp.sum(clustering_result[img_idx] == 0) == 0:
            print('\nSomehow KMeans seems failed. Try again\n')
            weighted_kmeans(
                superpixels, superpixel_features, superpixel_weights, k,
                n_superpixels_per_image)

    return clustering_result, xp.asarray(clustering_result == 0)


def superpixel_align(
        img, feature_map, superpixels, n_select=10, n_neighbor=4,
        append_pos=False):
    img_h = img.shape[1]
    feature_map_h, feature_map_w = feature_map.shape[1:]
    feature_ratio = float(feature_map_h) / img_h
    if isinstance(feature_map, chainer.Variable):
        feature_map = feature_map.array
    xp = cuda.get_array_module(feature_map)
    yy, xx = xp.meshgrid(xp.arange(feature_map_h), xp.arange(feature_map_w))
    ft_coords = xp.stack([yy, xx]).transpose(1, 2, 0) + 0.5
    flat_ft_coords = ft_coords.reshape(-1, 2)

    superpixel_features = []
    idxes = np.unique(superpixels)
    superpixels = xp.asarray(superpixels)
    for idx in np.sort(idxes):
        mask = superpixels == idx
        if append_pos:
            centroid = measurements.center_of_mass(cuda.to_cpu(mask))
        y, x = xp.where(mask)
        inside_coords = list(zip(y.tolist(), x.tolist()))
        random.shuffle(inside_coords)
        inside_coords = xp.asarray(inside_coords, dtype=np.float)
        selected_points = inside_coords[:n_select]
        selected_points *= feature_ratio
        selected_points += 0.5  # Use center of pixels
        selected_points[:, 0] = xp.clip(
            selected_points[:, 0], 0, feature_map_h - 1 + 0.5)
        selected_points[:, 1] = xp.clip(
            selected_points[:, 1], 0, feature_map_w - 1 + 0.5)
        features_in_sp = []
        for p in selected_points:
            py, px = p
            dist = xp.sqrt(((flat_ft_coords - p[None, :]) ** 2).sum(axis=1))
            idx = xp.argsort(dist)[:n_neighbor]
            neighbor_ft_coords = flat_ft_coords[idx]
            max_y, max_x = xp.max(neighbor_ft_coords, axis=0)
            min_y, min_x = xp.min(neighbor_ft_coords, axis=0)
            assert max_x > min_x, \
                '{} <= {}, \nidx:{}, \nneighbor_ft_coords:\n{}, \np:{}'.format(
                    max_x, min_x, idx, neighbor_ft_coords, p)
            assert max_y > min_y, \
                '{} <= {}, \nidx:{}, \nneighbor_ft_coords:\n{}, \np:{}'.format(
                    max_y, min_y, idx, neighbor_ft_coords, p)

            # Bilinear interpolation
            f11 = feature_map[:, int(min_y), int(min_x)]
            f12 = feature_map[:, int(max_y), int(min_x)]
            f21 = feature_map[:, int(min_y), int(max_x)]
            f22 = feature_map[:, int(max_y), int(max_x)]

            fp = (max_x - px) * (max_y - py) * f11
            fp += (max_x - px) * (py - min_y) * f12
            fp += (px - min_x) * (max_y - py) * f21
            fp += (px - min_x) * (py - min_y) * f22
            fp = 1. / ((max_x - min_x) * (max_y - min_y)) * fp

            # Add the coordinate of center of mas to the feature vector
            if append_pos:
                fp = xp.hstack([fp, xp.array(centroid)])

            features_in_sp.append(fp)
        features_in_sp = xp.stack(features_in_sp)
        superpixel_features.append(xp.mean(features_in_sp, axis=0))

    return xp.stack(superpixel_features)


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


def batch_superpixel_align(args, model, imgs, superpixels, feature_maps):
    xp = model.xp
    superpixel_features = None
    n_superpixels_per_image = []
    for img, superpixel, feature_map in zip(imgs, superpixels, feature_maps):
        n_superpixels_per_image.append(len(np.unique(superpixel)))
        feat = superpixel_align(
            img, feature_map, superpixel, args.n_anchors, args.n_neighbors,
            not args.without_pos)
        if superpixel_features is None:
            superpixel_features = feat
        else:
            superpixel_features = xp.concatenate(
                [superpixel_features, feat], axis=0)
    return superpixel_features, n_superpixels_per_image


def batch_create_prior(args, superpixels):
    superpixel_weights = None
    for superpixel in superpixels:
        prior = create_prior(
            superpixel, args.y_rel_pos, args.x_rel_pos, args.y_rel_sigma,
            args.x_rel_sigma)
        if superpixel_weights is None:
            superpixel_weights = prior
        else:
            superpixel_weights = np.concatenate(
                [superpixel_weights, prior], axis=0)
    return superpixel_weights


def batch_weighted_kmeans(args, superpixels, superpixel_features,
                          superpixel_weights, n_superpixels_per_image):
    superpixels = cuda.to_gpu(superpixels, args.gpu)
    superpixel_features = cuda.to_gpu(superpixel_features, args.gpu)
    superpixel_weights = cuda.to_gpu(superpixel_weights, args.gpu)
    clustering_results, road_masks = weighted_kmeans(
        superpixels, superpixel_features, superpixel_weights,
        args.n_clusters, n_superpixels_per_image)
    if args.gpu >= 0:
        clustering_results = cuda.to_cpu(clustering_results)
        road_masks = cuda.to_cpu(road_masks)
    return clustering_results, road_masks


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

    ret = evaluations.calc_semantic_segmentation_confusion(
        [road_mask], [label])
    TP = int(ret[1, 1])
    FP = int(ret[0, 1])
    FN = int(ret[1, 0])
    precision = float(TP / (TP + FP)) if TP + FP > 0 else None
    recall = float(TP / (TP + FN)) if TP + FN > 0 else None
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


def estimate_road_mask(imgs, img_fns, labels, label_fns, model, args):
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

    for img_fn, label_fn, clustering_result, road_mask in zip(
            img_fns, label_fns, clustering_results, road_masks):

        # Load image
        img = np.asarray(Image.open(img_fn), dtype=np.uint8)

        # Load labels
        label = np.asarray(Image.open(label_fn), dtype=np.uint8)
        label = create_label_mask(label.copy())

        if road_mask.shape != label.shape:
            h, w = label.shape
            road_mask = cv.resize(
                road_mask.astype(np.uint8), (w, h),
                interpolation=cv.INTER_NEAREST)
            clustering_result = cv.resize(
                clustering_result.astype(np.uint8), (w, h),
                interpolation=cv.INTER_NEAREST)

        save_image(args, img, road_mask, label, clustering_result, img_fn)
        result_info = save_info(img_fn, label_fn, road_mask, clustering_result,
                                label, elapsed_times, st_all)

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
