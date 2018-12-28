#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys  # NOQA isort:skip
sys.path.insert(0, '.')  # NOQA isort:skip

from chainercv import evaluations
import numpy as np
from tqdm import tqdm
import train_segnet

data_dir = 'data/cityscapes'
resize_shape = (1024, 2048)
dataset = train_segnet.CityscapesRoadDataset(data_dir, resize_shape)

road_ious = []
TPs = []
FPs = []
FNs = []
precisions = []
recalls = []
for _, label in tqdm(dataset):
    pred = np.zeros_like(label).astype(np.int32)
    pred[pred.shape[0] // 2:] = 1
    ret = evaluations.calc_semantic_segmentation_confusion([pred], [label])
    TP = ret[1, 1]
    FP = ret[0, 1]
    FN = ret[1, 0]
    TPs.append(TP)
    FPs.append(FP)
    FNs.append(FN)
    precisions.append(TP / (TP + FP))
    recalls.append(TP / (TP + FN))
    ret = evaluations.calc_semantic_segmentation_iou(ret)
    road_iou = ret[1]
    road_ious.append(road_iou)

print('Road IoU:', np.mean(road_ious))
print('Precision:', np.sum(TPs) / (np.sum(TPs) + np.sum(FPs)))
print('Average precision:', np.nanmean(precisions))
print('Recall:', np.sum(TPs) / (np.sum(TPs) + np.sum(FNs)))
print('Average recall:', np.nanmean(recalls))
