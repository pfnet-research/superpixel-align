#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from collections import defaultdict
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='results')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    experiment_logs = {}
    result_dirs = glob.glob(os.path.join(args.result_dir, '*'))
    for result_dir in result_dirs:
        if not os.path.basename(result_dir).startswith('train_'):
            continue
        logs = glob.glob(os.path.join(result_dir, 'log'))
        logs += glob.glob(os.path.join(result_dir, '*', 'log'))
        if len(logs) == 0:
            continue
        latest_log = json.load(open(logs[0]))
        for log in logs[1:]:
            log = json.load(open(log))
            if latest_log[-1]['iteration'] < log[-1]['iteration']:
                latest_log = log

        if latest_log[-1]['iteration'] < 2000:
            continue

        next_round_dirs = []
        for fn in glob.glob(os.path.join(result_dir, '*')):
            if not os.path.isdir(fn):
                continue
            if not os.path.basename(fn).startswith('train_'):
                continue
            if 'eval' in fn:
                continue
            next_round_dirs.append(fn)
        if len(next_round_dirs) == 0:
            next_round_dirs.append(result_dir)
        for dname in sorted(next_round_dirs):
            if os.path.exists(os.path.join(dname, 'args.txt')):
                latest_result_dir = dname
        experiment_logs[latest_result_dir] = latest_log

    fig0, fig1 = plt.figure(), plt.figure()
    ax0, ax1 = fig0.add_subplot(1, 1, 1), fig1.add_subplot(1, 1, 1)

    after_rounds = defaultdict(list)
    sorted_logs = sorted([(np.max([l['val/main/iou/road'] for l in v]), (k, v))
                          for k, v in experiment_logs.items()], reverse=True)
    print('Max road IoU,Result dir,Use MSE, Use soft label,'
          '1 round,2 round,3 round')
    for max_iou, (key, value) in sorted_logs:
        train_args = json.load(open(os.path.join(key, 'args.txt')))
        if 'use_mse' in train_args:
            use_soft_label = train_args['use_soft_label']
            use_mse = train_args['use_mse']
        else:
            use_soft_label = False
            use_mse = train_args['use_soft_label']

        iters = [v['iteration'] for v in value]
        ious = [v['val/main/iou/road'] for v in value]
        label = '{:.3f} {}'.format(max_iou, key)

        dname = os.path.dirname(key)
        key = dname if not dname.endswith('results') else key
        top_train_args = json.load(open(os.path.join(key, 'args.txt')))
        if key.startswith('/'):
            key = '{}:{}'.format(os.uname().nodename, key)
        print('{},{},{},{},'.format(
            max_iou, key, use_mse, use_soft_label), end='')
        for v in value:
            if v['iteration'] % int(top_train_args['train_limit'][0]) == 0:
                after_rounds[key].append(v['val/main/iou/road'])
                print('{},'.format(v['val/main/iou/road']), end='')
        if len(after_rounds[key]) == 0:
            print('{},'.format(value[-1]['val/main/iou/road']), end='')
        print('')

        ax0.plot(iters, ious, label=label)
        ax1.plot(after_rounds[key], label=label)

    ax0.legend(loc=(1.1, 0))
    ax1.legend(loc=(1.1, 0))

    fig0.savefig('iou_logs_iter.pdf', dpi=300, bbox_inches='tight')
    fig1.savefig('iou_logs_round.pdf', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
