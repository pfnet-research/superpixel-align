#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('result_json', type=str)
parser.add_argument('--show_failed_fn', action='store_true', default=False)
parser.add_argument('--count_duplicated', action='store_true', default=False)
parser.add_argument('--n_imgs', type=int, default=None)
args = parser.parse_args()

checked = {}

msg = ''
road_iou = []
non_road_iou = []
precisions = []
recalls = []
TPs = []
FPs = []
FNs = []


def collect(data):
    if data['road_iou'] == 0:
        if args.show_failed_fn:
            print(data['img_fn'])
    road_iou.append(data['road_iou'])
    non_road_iou.append(data['non_road_iou'])
    TPs.append(data['TP'])
    FPs.append(data['FP'])
    FNs.append(data['FN'])
    if data['precision']:
        precisions.append(data['precision'])
    else:
        precisions.append(np.nan)
    if data['recall']:
        recalls.append(data['recall'])
    else:
        recalls.append(np.nan)


for line in open(args.result_json):
    data = json.loads(line.strip())

    if not args.count_duplicated:
        if data['img_fn'] not in checked:
            checked[data['img_fn']] = data['road_iou']
        else:
            # If already exists, skip it
            continue

    collect(data)

if args.n_imgs is not None:
    road_iou = road_iou[:args.n_imgs]
    non_road_iou = non_road_iou[:args.n_imgs]
    precisions = precisions[:args.n_imgs]
    recalls = recalls[:args.n_imgs]
    TPs = TPs[:args.n_imgs]
    FPs = FPs[:args.n_imgs]
    FNs = FNs[:args.n_imgs]

msg += 'Road mean IoU\t:{}\n'.format(np.nanmean(road_iou))
msg += 'Road min IoU\t:{}\n'.format(np.nanmin(road_iou))
msg += 'Road max IoU\t:{}\n'.format(np.nanmax(road_iou))

msg += 'Non-road mean IoU\t:{}\n'.format(np.nanmean(non_road_iou))
msg += 'Non-road min IoU\t:{}\n'.format(np.nanmin(non_road_iou))
msg += 'Non-road max IoU\t:{}\n'.format(np.nanmax(non_road_iou))

msg += 'Average Precision\t:{}\n'.format(np.nanmean(precisions))
msg += 'Precision\t:{}\n'.format(np.sum(TPs) / (np.sum(TPs) + np.sum(FPs)))
msg += 'Min Precision\t:{}\n'.format(np.nanmin(precisions))
msg += 'Max Precision\t:{}\n'.format(np.nanmax(precisions))
msg += 'N\t:{}\n'.format(len(precisions))

msg += 'Average Recall\t:{}\n'.format(np.nanmean(recalls))
msg += 'Recall\t:{}\n'.format(np.sum(TPs) / (np.sum(TPs) + np.sum(FNs)))
msg += 'Min Recall\t:{}\n'.format(np.nanmin(recalls))
msg += 'Max Recall\t:{}\n'.format(np.nanmax(recalls))
msg += 'N\t:{}\n'.format(len(recalls))

msg += '\n'
for fn, iou in sorted(checked.items(), key=lambda x: x[1], reverse=True)[:10]:
    msg += '{}\t{}\n'.format(iou, fn)

print(args.result_json)
print(msg)

with open(os.path.join(
        os.path.dirname(args.result_json), 'summary.txt'), 'w') as fp:
    print(msg, file=fp)
