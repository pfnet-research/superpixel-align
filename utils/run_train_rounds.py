#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys  # NOQA isort:skip
sys.path.insert(0, '.')  # NOQA isort:skip

import argparse
import glob
import math
from multiprocessing import Manager
from multiprocessing import Pool
from multiprocessing import Queue
import os
import subprocess
import time
import zipfile

from labels_from_segnet import save_labels
import numpy as np
from tqdm import tqdm
from train_segnet import create_result_dir

pred_score_queue = Queue()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_round', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=2000)
    parser.add_argument('--val_iteration', type=int, default=100)
    parser.add_argument('--n_use_data', type=int, default=None)
    parser.add_argument('--use_soft_label', action='store_true', default=False)
    parser.add_argument('--use_mse', action='store_true', default=False)
    parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument('--test_mode', action='store_true', default=False)
    parser.add_argument('--save_each', action='store_true', default=False)
    parser.add_argument('--n_gpus', type=int, default=8)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--result_base_dir', type=str, default='results')
    parser.add_argument('--resume_round', type=int, default=2,
                        help='The default is 2')
    parser.add_argument('--first_result_dir', type=str, default=None,
                        help='For resuming')
    parser.add_argument('--out_zip_fn', type=str, default=None,
                        help='For resuming')
    parser.add_argument('--eval_shape', type=int,
                        nargs=2, default=[1024, 2048])
    parser.add_argument('--img_zip_fn', type=str,
                        default='data/cityscapes_train_imgs.0.zip')
    parser.add_argument('--label_zip_fn', type=str,
                        default='data/cityscapes_train_labels.0.zip')
    parser.add_argument('--estimated_label_zip_fn', type=str,
                        default='results/estimated_train_labels.0.zip')
    args = parser.parse_args()

    if args.test_mode:
        args.iteration = 10
        args.val_iteration = 10
        args.n_labels = 16
        args.n_use_data = 16
        args.n_round = 3
    elif 'train_extra' in args.img_zip_fn:
        args.n_labels = 22973
    else:
        args.n_labels = 2975

    return args


def start_first_round(
        iteration, val_iteration, n_use_data, random, test_mode,
        img_zip_fn, estimated_label_zip_fn, result_base_dir, n_gpus,
        batchsize):
    if test_mode:
        result_dir = create_result_dir('{}/Trash/train_round1'.format(
            result_base_dir))
    elif 'train_extra' in img_zip_fn:
        result_dir = create_result_dir('{}/train_extra_round1'.format(
            result_base_dir))
    else:
        result_dir = create_result_dir('{}/train_round1'.format(
            result_base_dir))
    cmd = 'MPLBACKEND=Agg PYTHONWARNINGS=ignore PYTHONOPTIMIZE=1 '
    cmd += 'mpiexec -np {} '.format(args.n_gpus)
    cmd += '-x PATH '
    cmd += '-x LIBRARY_PATH '
    cmd += '-x LD_LIBRARY_PATH '
    cmd += '-x CPATH '
    cmd += '-x MPLBACKEND '
    cmd += 'python -O train_segnet.py '
    cmd += '--model basic '
    cmd += '--optimizer Adam '
    cmd += '--train_limit {iteration} iteration '
    cmd += '--val_interval {val_iteration} iteration '
    cmd += '--log_interval {val_iteration} iteration '
    cmd += '--batchsize {} '.format(batchsize)
    cmd += '--input_shape 512 1024 '
    cmd += '--train_img_zip {} '.format(img_zip_fn)
    cmd += '--train_label_zip {} '.format(estimated_label_zip_fn)
    cmd += '--val_img_zip data/cityscapes_val_imgs.0.zip '
    cmd += '--val_label_zip data/cityscapes_val_labels.0.zip '
    cmd += '--result_dir {result_dir} '

    if n_use_data is not None:
        cmd += '--n_use_data {} '.format(n_use_data)

    if random:
        cmd += '--random '

    cmd = cmd.format(iteration=iteration, val_iteration=val_iteration,
                     result_dir=result_dir)

    print('train_img_zip:', img_zip_fn)
    print('train_label_zip:', estimated_label_zip_fn)
    print('val_img_zip: data/cityscapes_val_imgs.0.zip')
    print('val_label_zip: data/cityscapes_val_labels.0.zip')
    print('-' * 20)
    print(cmd)
    print('=' * 20)
    subprocess.run(cmd, shell=True)
    return result_dir


def start_next_round(
        first_result_dir, resume_dir, resume_iteration, end_iteration,
        val_iteration, label_zip, n_round, use_soft_label, use_mse, n_use_data,
        random, train_img_zip, n_gpus, batchsize):
    assert n_round >= 2, 'round should be greater than 2'
    if 'train_extra' in train_img_zip:
        result_dir = create_result_dir(
            '{}/train_extra_round{}'.format(first_result_dir, n_round))
    else:
        result_dir = create_result_dir('{}/train_round{}'.format(
            first_result_dir, n_round))
    cmd = 'MPLBACKEND=Agg PYTHONWARNINGS=ignore PYTHONOPTIMIZE=1 '
    cmd += 'mpiexec -np {} '.format(n_gpus)
    cmd += '-x PATH '
    cmd += '-x LIBRARY_PATH '
    cmd += '-x LD_LIBRARY_PATH '
    cmd += '-x CPATH '
    cmd += '-x MPLBACKEND '
    cmd += 'python -O train_segnet.py '
    cmd += '--model basic '
    cmd += '--optimizer Adam '
    cmd += '--train_limit {end_iteration} iteration '
    cmd += '--val_interval {val_iteration} iteration '
    cmd += '--log_interval {val_iteration} iteration '
    cmd += '--batchsize {} '.format(batchsize)
    cmd += '--input_shape 512 1024 '
    cmd += '--train_img_zip {} '.format(train_img_zip)
    cmd += '--train_label_zip {label_zip} '
    cmd += '--val_img_zip data/cityscapes_val_imgs.0.zip '
    cmd += '--val_label_zip data/cityscapes_val_labels.0.zip '
    cmd += '--result_dir {result_dir} '
    cmd += '--resume {resume_dir}/snapshot_iter_{resume_iteration} '

    if use_soft_label:
        cmd += '--use_soft_label '
    elif use_mse:
        cmd += '--use_mse '

    if n_use_data is not None:
        cmd += '--n_use_data {} '.format(n_use_data)

    if random:
        cmd += '--random'

    cmd = cmd.format(
        label_zip=label_zip, result_dir=result_dir, n_round=n_round,
        resume_dir=resume_dir, end_iteration=end_iteration,
        resume_iteration=resume_iteration, val_iteration=val_iteration)

    print(cmd)
    subprocess.run(cmd, shell=True)
    return result_dir


def run_labels_from_segnet(
        gpu_id, param_dir, iteration, split, first_result_dir, img_zip_fn,
        label_zip_fn, start_index, end_index, soft_label, eval_shape,
        save_each):
    out_dir = '{}/iter-{}_eval-{}'.format(first_result_dir, iteration, split)
    pred_and_scores = save_labels(
        param_dir, iteration, gpu_id, img_zip_fn, label_zip_fn, out_dir,
        start_index, end_index, soft_label, eval_shape, save_each)
    if not save_each:
        for key, value in pred_and_scores.items():
            pred_score_queue.put((key, value))


def write_worker(out_zip_fn, n_labels):
    i = 0
    d = {}
    while True:
        ret = pred_score_queue.get()
        key, value = ret
        d.update({key: value})
        if len(d) == n_labels * 2:
            break
        i += 1
    fp = open(out_zip_fn, 'wb')
    np.savez(fp, **d)
    fp.close()


def create_label_from_model(
        param_dir, iteration, split, first_result_dir, use_soft_label, use_mse,
        eval_shape, n_labels, n_gpus, img_zip_fn, label_zip_fn, save_each):
    out_zip_fn = '{}/iter-{}_eval-{}.0.zip'.format(
        first_result_dir, iteration, split)

    # Whether save soft labels or not
    soft_label = use_soft_label or use_mse

    args = []
    step = math.ceil(n_labels / n_gpus)
    gpu_id = 0
    for i in range(0, n_labels, step):
        if i + step >= n_labels:
            end_i = n_labels
        else:
            end_i = i + step
        args.append(
            (gpu_id, param_dir, iteration, split, first_result_dir,
             img_zip_fn, label_zip_fn, i, end_i, soft_label, eval_shape,
             save_each))
        gpu_id += 1

    with Pool(n_gpus + 1) as p:
        if not save_each:
            writer_ret = p.apply_async(write_worker, (out_zip_fn, n_labels))
        ret = [p.apply_async(run_labels_from_segnet, args=arg) for arg in args]
        for r in ret:
            r.get()
        if not save_each:
            writer_ret.get()

    if save_each:
        print('zipping files...')
        out_dir = '{}/iter-{}_eval-{}'.format(
            first_result_dir, iteration, split)
        zf = zipfile.ZipFile(out_dir + '.0.zip', 'w')
        for fn in tqdm(glob.glob(os.path.join(out_dir, '*.npy'))):
            zf.write(fn)
        zf.close()

    return out_zip_fn


if __name__ == '__main__':
    args = get_args()
    split = 'train_extra' if 'train_extra' in args.img_zip_fn else 'train'

    if args.first_result_dir is None:
        first_result_dir = start_first_round(
            args.iteration, args.val_iteration, args.n_use_data, args.random,
            args.test_mode, args.img_zip_fn, args.estimated_label_zip_fn,
            args.result_base_dir, args.n_gpus, args.batchsize)
    else:
        first_result_dir = args.first_result_dir

    if args.out_zip_fn is None:
        out_zip_fn = create_label_from_model(
            first_result_dir, args.iteration, split, first_result_dir,
            args.use_soft_label, args.use_mse, args.eval_shape, args.n_labels,
            args.n_gpus, args.img_zip_fn, args.label_zip_fn, args.save_each)
    else:
        out_zip_fn = args.out_zip_fn

    print('First round finished')
    print('result_dir:', first_result_dir)
    print('out_zip_fn:', out_zip_fn)
    print('-' * 20)

    prev_result_dir = first_result_dir
    end_iteration = args.iteration
    for n_round in range(args.resume_round, args.n_round + 1):
        resume_iteration = end_iteration
        end_iteration = args.iteration * n_round

        prev_result_dir = start_next_round(
            first_result_dir, prev_result_dir, resume_iteration, end_iteration,
            args.val_iteration, out_zip_fn, n_round, args.use_soft_label,
            args.use_mse, args.n_use_data, args.random, args.img_zip_fn,
            args.n_gpus, args.batchsize)

        out_zip_fn = create_label_from_model(
            prev_result_dir, end_iteration, split, first_result_dir,
            args.use_soft_label, args.use_mse, args.eval_shape, args.n_labels,
            args.n_gpus, args.img_zip_fn, args.label_zip_fn, args.save_each)

        print('{}th round finished'.format(n_round))
        print('result_dir:', prev_result_dir)
        print('out_zip_fn:', out_zip_fn)
        print('-' * 20)
