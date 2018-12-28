#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import glob
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--job-id-list', type=str, default=None)
args = parser.parse_args()


def main():
    out_dir = os.path.join('results', 'experiments2')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if args.job_id_list is None:
        cmd = 'dmux jobs --limit 10000 | grep ablation_study'
        ret = [line.strip().split() for line in subprocess.check_output(cmd, shell=True).decode('utf-8').split('\n')]
    else:
        job_ids = [job_id.strip() for job_id in open(args.job_id_list).readlines()]
        job_ids = ' -e '.join(job_ids)
        cmd = 'dmux jobs --limit 10000 | grep -e ' + job_ids
        ret = [line.strip().split() for line in subprocess.check_output(cmd, shell=True).decode('utf-8').split('\n')]

    for line in ret:
        if not line:
            continue
        job_id = line[0]
        n_clusters, batchsize, granularity = line[10:13]
        dirname = 'ncluster-{}_batchsize-{}_granularity-{}'.format(n_clusters, batchsize, granularity)

        experiment_dir = os.path.join(out_dir, dirname)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        src_dir = os.path.join('results', job_id)
        dst_dir = os.path.join(out_dir, dirname, job_id)
        shutil.copytree(src_dir, dst_dir)
        print(src_dir, dirname)


if __name__ == '__main__':
    main()
