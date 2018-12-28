#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import glob
import json
import os

import numpy as np

result_dirs = [
    'results/experiments/ncluster-2_batchsize-30_granularity-300',
    'results/experiments/ncluster-3_batchsize-30_granularity-300',
    'results/experiments/ncluster-4_batchsize-10_granularity-300',
    'results/experiments/ncluster-4_batchsize-1_granularity-300',
    'results/experiments/ncluster-4_batchsize-20_granularity-300',
    'results/experiments/ncluster-4_batchsize-30_granularity-100',
    'results/experiments/ncluster-4_batchsize-30_granularity-200',
    'results/experiments/ncluster-4_batchsize-30_granularity-300',
    'results/experiments/ncluster-4_batchsize-30_granularity-400',
    'results/experiments/ncluster-4_batchsize-30_granularity-500',
    'results/experiments/ncluster-4_batchsize-30_granularity-600',
    'results/experiments/ncluster-4_batchsize-30_granularity-700',
    'results/experiments/ncluster-4_batchsize-30_granularity-800',
    'results/experiments/ncluster-4_batchsize-40_granularity-300',
    'results/experiments/ncluster-4_batchsize-50_granularity-300',
    'results/experiments/ncluster-5_batchsize-30_granularity-300',
    'results/experiments/ncluster-6_batchsize-30_granularity-300',
    'results/experiments/ncluster-7_batchsize-30_granularity-300',
    'results/experiments/ncluster-8_batchsize-30_granularity-300',
]

result_dirs = ['results/experiments/raw-CNN_clustering_spoverlap']

# summary_dname = 'estimated_train_all_labels'
# summary_dname = 'estimate_labels_using_direct_feature_clustering'
summary_dname = 'estimate_labels_using_direct_feature_clustering'


def calc_mean_iou(result_dir):
    miou = []
    results = []
    for dname in glob.glob(os.path.join(result_dir, '*')):
        if not os.path.isdir(dname):
            continue
        fn = os.path.join(dname, summary_dname, 'result.json')
        for line in open(fn):
            datum = json.loads(line.strip())
            if datum['road_iou'] > 0:
                results.append(datum)
                miou.append(datum['road_iou'])
    return np.nanmean(miou)


print('n_cluster,batchsize,granularity,mIoU')
for result_dir in result_dirs:
    miou = calc_mean_iou(result_dir)
    if 'ncluster' in result_dir:
        n_clusters = re.search('ncluster-([0-9])', result_dir).groups()[0]
        batchsize = re.search('batchsize-([0-9]+)', result_dir).groups()[0]
        granularity = re.search('granularity-([0-9]+)', result_dir).groups()[0]
        print('{},{},{},{}'.format(n_clusters, batchsize, granularity, miou))
    else:
        print('mIoU:', miou)
