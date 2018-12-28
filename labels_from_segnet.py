#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys  # NOQA isort:skip
sys.path.insert(0, 'datasets')  # NOQA isort:skip

import argparse
import glob
import json
import os
import zipfile

from PIL import Image
import chainer
from chainer import serializers
from chainercv import evaluations
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import train_segnet
from zipped_cityscapes_road_dataset import ZippedCityscapesRoadDataset

chainer.config.train = False


def save_labels(param_dir, iteration, gpu, img_zip_fn, label_zip_fn, out_dir,
                start_index, end_index, soft_label, eval_shape,
                save_each=False):
    train_args = json.load(open(os.path.join(param_dir, 'args.txt')))

    if not os.path.exists(out_dir):
        try:
            os.makedirs(out_dir)
        except Exception:
            pass

    # Find the target snapshot
    snapshots = sorted(glob.glob(os.path.join(param_dir, 'snapshot_*')))
    for snapshot in snapshots:
        if 'iter_{}'.format(iteration) in snapshot:
            break

    # Create model
    if train_args['model'] == 'basic':
        model = train_segnet.SegNetBasic(n_class=2, pred_shape=eval_shape)
    elif train_args['model'] == 'normal':
        model = train_segnet.SegNet(n_class=2)

    # Load model parameters
    serializers.load_npz(
        snapshot, model, path='updater/model:main/predictor/')
    if gpu >= 0:
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu(gpu)

    # Create dataset
    d = ZippedCityscapesRoadDataset(
        img_zip_fn, label_zip_fn, train_args['input_shape'])
    if end_index > len(d):
        raise ValueError(
            'end_index option should be less than the length of dataset '
            '{} but {} was given.'.format(len(d), end_index))

    if not save_each:
        pred_and_scores = {}
    for i in tqdm(range(start_index, end_index)):
        img, label = d[i]
        pred, score = model.predict([img], True)[0]
        assert pred.ndim == 2, pred.ndim
        assert pred.shape == tuple(eval_shape), \
            'pred:{} but eval_shape:{}'.format(pred.shape, eval_shape)
        assert score.ndim == 3, score.ndim
        assert score.shape[1:] == tuple(eval_shape), \
            'score[1:]:{} but eval_shape: {}'.format(
                score.shape[1:], eval_shape)

        # Evaluate prediction
        ret = evaluations.calc_semantic_segmentation_confusion([pred], [label])
        TP = int(ret[1, 1])
        FP = int(ret[0, 1])
        FN = int(ret[1, 0])
        precision = float(TP / (TP + FP)) if TP + FP > 0 else None
        recall = float(TP / (TP + FN)) if TP + FN > 0 else None
        iou = evaluations.calc_semantic_segmentation_iou(ret)

        pred = pred.astype(np.bool)
        score = score.astype(np.float32)
        fn_base = os.path.splitext(os.path.basename(d.img_fns[i]))[0]
        save_fn = os.path.join(out_dir, fn_base)
        if save_each:
            np.save(save_fn, pred)
            np.save(save_fn + '_scores', pred)
        else:
            pred_and_scores[save_fn] = pred
            pred_and_scores[save_fn + '_scores'] = score

        plt.clf()
        fig, axes = plt.subplots(1, 3)
        fig.set_dpi(300)
        axes[0].axis('off')
        axes[1].axis('off')
        axes[2].axis('off')

        # Show result
        img = np.array(Image.open(d.img_fns[i]), dtype=np.uint8)
        axes[0].imshow(img)
        axes[0].imshow(pred, alpha=0.4, cmap=plt.cm.Set1_r)
        axes[0].set_title('Estimated road mask (input image overlayed)',
                          fontsize=4)
        # Show labels
        axes[1].imshow(label == 1)
        axes[1].set_title('Ground truth road mask', fontsize=4)
        # Show road estimation
        axes[2].imshow(pred)
        axes[2].set_title('Estimated road mask', fontsize=4)

        plt.savefig(os.path.join(out_dir, os.path.basename(d.img_fns[i])),
                    bbox_inches='tight')
        plt.close()

        with open(os.path.join(out_dir, 'result.json'), 'a') as fp:
            result_info = {
                'img_fn': d.img_fns[i],
                'label_fn': d.label_fns[i],
                'road_iou': iou[1],
                'non_road_iou': iou[0],
                'precision': precision,
                'recall': recall,
                'TP': TP,
                'FP': FP,
                'FN': FN
            }
            result_info.update({
                'param_dir': param_dir,
                'iteration': iteration,
                'gpu': gpu,
                'img_zip_fn': img_zip_fn,
                'label_zip_fn': label_zip_fn,
                'out_dir': out_dir,
                'start_index': start_index,
                'end_index': end_index,
                'soft_label': soft_label,
                'eval_shape': eval_shape,
                'save_each': save_each,
            })
            result_info.update({'train_args': train_args})
            print(json.dumps(result_info), file=fp)

    chainer.cuda.memory_pool.free_all_blocks()
    del model

    if not save_each:
        return pred_and_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--param_dir', type=str)
    parser.add_argument('--iteration', type=int)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--img_zip_fn', type=str)
    parser.add_argument('--label_zip_fn', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--start_index', type=int)
    parser.add_argument('--end_index', type=int)
    parser.add_argument('--soft_label', action='store_true', default=False)
    parser.add_argument(
        '--eval_shape', type=int, nargs=2, default=[1024, 2048])
    args = parser.parse_args()

    save_labels(
        args.param_dir, args.iteration, args.gpu, args.img_zip_fn,
        args.label_zip_fn, args.out_dir, args.start_index, args.end_index,
        args.soft_label, args.eval_shape, True)
