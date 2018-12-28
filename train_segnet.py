#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys  # NOQA isort:skip
sys.path.insert(0, 'datasets')  # NOQA isort:skip
sys.path.insert(0, 'models')  # NOQA isort:skip

import argparse
import copy
import json
import multiprocessing
import os
import random
import re
import shutil
import time

import chainer
from chainer import iterators
from chainer import optimizers
from chainer import reporter
from chainer import training
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainercv.evaluations import calc_semantic_segmentation_confusion
from chainercv.extensions import SemanticSegmentationEvaluator
from chainercv.utils import apply_prediction_to_iterator
import chainermn
from cityscapes_road_dataset import CityscapesRoadDataset
import cupy as cp
from estimated_cityscapes_dataset import EstimatedCityscapesDataset
import numpy as np
from segnet import SegNet
from segnet_basic import SegNetBasic
from zipped_cityscapes_road_dataset import ZippedCityscapesRoadDataset
from zipped_estimated_cityscapes_dataset import \
    ZippedEstimatedCityscapesDataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train_img_zip', type=str,
        default='data/cityscapes_train_imgs.0.zip',
        help='If it\'s given, ZippedEstimatedCityscapesDataset will be used.')
    parser.add_argument(
        '--train_label_zip', type=str,
        default='results/estimated_train_labels.0.zip',
        help='If it\'s given, ZippedEstimatedCityscapesDataset will be used.')
    parser.add_argument(
        '--val_img_zip', type=str,
        default='data/cityscapes_val_imgs.0.zip',
        help='If it\'s given, ZippedCityscapesRoadDataset will be used.')
    parser.add_argument(
        '--val_label_zip', type=str,
        default='data/cityscapes_gtFine_val_labels.0.zip',
        help='If it\'s given, ZippedCityscapesRoadDataset will be used.')

    parser.add_argument('--model', type=str, default='basic',
                        choices=['normal', 'basic'])
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--decay_iteration', type=int, default=300)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument(
        '--train_limit', type=str, nargs=2, default=['1000', 'iteration'])
    parser.add_argument('--optimizer', type=str, default='MomentumSGD',
                        choices=['Adam', 'MomentumSGD'])
    parser.add_argument(
        '--input_shape', type=int, nargs=2, default=[512, 1024])
    parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument('--communicator', type=str, default='single_node')
    parser.add_argument('--prefix', type=str, default='results/round_1')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument(
        '--log_interval', type=str, nargs=2, default=['50', 'iteration'])
    parser.add_argument(
        '--val_interval', type=str, nargs=2, default=['50', 'iteration'])
    parser.add_argument(
        '--eval_shape', type=int, nargs=2, default=[1024, 2048])
    parser.add_argument('--result_dir', type=str, default=None)
    parser.add_argument(
        '--use_soft_label', action='store_true', default=False,
        help='If True, softmax cross entorpy with soft labels is used as loss'
             'function')
    parser.add_argument(
        '--use_mse', action='store_true', default=False,
        help='If True, mean squared error is used as loss function')
    parser.add_argument('--n_use_data', type=int, default=None)

    args = parser.parse_args()
    return args


def create_result_dir(prefix):
    result_dir = '{}_{}_0'.format(
        prefix, time.strftime('%Y-%m-%d_%H-%M-%S'))
    while os.path.exists(result_dir):
        i = result_dir.split('_')[-1]
        result_dir = re.sub('_[0-9]+$', result_dir, '_{}'.format(i))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    shutil.copy(__file__, os.path.join(result_dir, os.path.basename(__file__)))
    return result_dir


class PrecisionRecallEvaluator(extensions.Evaluator):

    trigger = 1, 'epoch'
    default_name = 'val'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, iterator, target):
        super().__init__(iterator, target)

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        imgs, pred_values, gt_values = apply_prediction_to_iterator(
            target.predict, it)
        # delete unused iterator explicitly
        del imgs

        pred_labels, = pred_values
        gt_labels, = gt_values

        ret = calc_semantic_segmentation_confusion(pred_labels, gt_labels)

        report = {
            'FP': ret[0, 1],
            'FN': ret[1, 0],
            'precision': ret[1, 1] / (ret[1, 1] + ret[0, 1]),
            'recall': ret[1, 1] / (ret[1, 1] + ret[1, 0]),
        }

        observation = dict()
        with reporter.report_scope(observation):
            reporter.report(report, target)
        return observation


if __name__ == '__main__':
    args = get_args()

    # Prepare ChainerMN communicator.
    comm = chainermn.create_communicator(args.communicator)
    device = comm.intra_rank

    # Set seeds
    random.seed(comm.intra_rank)
    np.random.seed(comm.intra_rank)
    cp.random.seed(comm.intra_rank)

    if comm.mpi_comm.rank == 0:
        print(json.dumps(vars(args), indent=4, sort_keys=True))

    if comm.mpi_comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        print('Using {} communicator'.format(args.communicator))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Train finishes after: {}'.format(args.train_limit))
        print('==========================================')

    # Wheter the soft label is needed or not
    soft_label = args.use_soft_label or args.use_mse

    # Create datasets
    train = ZippedEstimatedCityscapesDataset(
        args.train_img_zip, args.train_label_zip, args.input_shape,
        args.random, soft_label)
    if args.n_use_data is not None:
        train = train[:args.n_use_data]
    if comm.mpi_comm.rank == 0:
        print('train dataset:', len(train))
    train = chainermn.scatter_dataset(train, comm, shuffle=True)

    # Only images are resized
    valid = ZippedCityscapesRoadDataset(
        args.val_img_zip, args.val_label_zip, args.input_shape)
    if comm.mpi_comm.rank == 0:
        print('valid dataset:', len(valid))
    valid = chainermn.scatter_dataset(valid, comm)

    # Change multiprocessing settings
    multiprocessing.set_start_method('forkserver')

    # Create iterator
    train_iter = iterators.MultithreadIterator(train, args.batchsize)
    valid_iter = iterators.MultithreadIterator(
        valid, args.batchsize, False, False)

    # Create model
    if args.model == 'normal':
        model = SegNet(n_class=2, comm=comm)
    elif args.model == 'basic':
        model = SegNetBasic(n_class=2, comm=comm, pred_shape=args.eval_shape)

    # Select loss function
    if args.use_soft_label:
        def softmax_cross_entropy_with_soft_labels(y, t):
            y = F.log_softmax(y)
            return -F.average(t * y)
        lossfun = softmax_cross_entropy_with_soft_labels
        if comm.mpi_comm.rank == 0:
            print('lossfun:', 'softmax_cross_entropy_with_soft_labels')
    elif args.use_mse:
        lossfun = F.mean_squared_error
        if comm.mpi_comm.rank == 0:
            print('lossfun:', 'F.mean_squared_error')
    else:
        lossfun = F.softmax_cross_entropy
        if comm.mpi_comm.rank == 0:
            print('lossfun:', 'F.softmax_cross_entropy')

    model = L.Classifier(model, lossfun)
    model.compute_accuracy = False
    chainer.cuda.get_device(device).use()  # Make the GPU current
    model.to_gpu()

    if args.optimizer == 'Adam':
        optimizer = optimizers.Adam()
        optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
        optimizer.setup(model)
    elif args.optimizer == 'MomentumSGD':
        optimizer = optimizers.MomentumSGD(args.lr)
        optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
        optimizer.setup(model)
        if args.weight_decay > 0:
            optimizer.add_hook(
                chainer.optimizer.WeightDecay(args.weight_decay))

    updater = training.StandardUpdater(train_iter, optimizer, device=device)

    if comm.mpi_comm.rank == 0:
        if args.result_dir is None:
            result_dir = create_result_dir(args.prefix)
        else:
            result_dir = args.result_dir
    else:
        import tempfile
        result_dir = tempfile.mkdtemp(dir='/tmp/')

    json.dump(vars(args), open(os.path.join(result_dir, 'args.txt'), 'w'),
              indent=4, sort_keys=True)

    trainer = training.Trainer(
        updater, (int(args.train_limit[0]), args.train_limit[1]),
        out=result_dir)

    if args.optimizer == 'MomentumSGD' and args.decay_iteration > 0:
        trainer.extend(
            extensions.ExponentialShift('lr', 0.1, optimizer=optimizer),
            trigger=(args.decay_iteration, 'iteration'))

    log_interval = (int(args.log_interval[0]), args.log_interval[1])
    val_interval = (int(args.val_interval[0]), args.val_interval[1])

    evaluator = SemanticSegmentationEvaluator(
        valid_iter, model.predictor, ['non_road', 'road'])
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    pr_evaluator = PrecisionRecallEvaluator(valid_iter, model.predictor)
    pr_evaluator = chainermn.create_multi_node_evaluator(pr_evaluator, comm)

    trainer.extend(evaluator, trigger=val_interval, name='val')
    trainer.extend(pr_evaluator, trigger=val_interval, name='val_')
    trainer.extend(extensions.LogReport(trigger=log_interval))

    if comm.rank == 0:
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.snapshot(
            filename='snapshot_iter_{.updater.iteration}'),
            trigger=val_interval)
        entries = ['iteration', 'main/loss', 'val/main/iou/road',
                   'val_/main/precision', 'val_/main/recall', 'lr',
                   'elapsed_time']
        logger = extensions.LogReport(trigger=log_interval)
        trainer.extend(extensions.PrintReport(entries, logger),
                       trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'val/main/loss'],
            'iteration', file_name='loss.png', trigger=log_interval))
        trainer.extend(extensions.PlotReport(
            ['val/main/class_accuracy/road',
             'val/main/class_accuracy/non_road'],
            'iteration', file_name='accuracy.png', trigger=log_interval))
        trainer.extend(extensions.PlotReport(
            ['val/main/iou/road', 'val/main/iou/non_road'],
            'iteration', file_name='ious.png', trigger=log_interval))
        trainer.extend(extensions.PlotReport(
            ['val_/main/precision', 'val_/main/recall'],
            'iteration', file_name='prerec.png', trigger=log_interval))

    if args.resume is not None:
        chainer.serializers.load_npz(args.resume, trainer, strict=False)

    trainer.run()
