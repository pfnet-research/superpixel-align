#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

import cv2 as cv
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--pred_label_dir', type=str)
parser.add_argument('--img_dir', type=str,
                    default='data/cityscapes/leftImg8bit/demoVideo')
parser.add_argument('--out_video_fn', type=str,
                    default='results/preds_labels.avi')
args = parser.parse_args()

alpha = 0.5
road_color = [128, 64, 128]
img_fns = sorted(glob.glob(os.path.join(args.img_dir, '*', '*.png')))
pred_fns = sorted(glob.glob(os.path.join(args.pred_label_dir, '*.png')))

fourcc = cv.VideoWriter_fourcc(*'MJPG')
out = cv.VideoWriter(args.out_video_fn, fourcc, 30.0, (2048, 1024))

for img_fn, pred_fn in tqdm(zip(img_fns, pred_fns)):
    img = cv.imread(img_fn)
    pred = cv.imread(pred_fn, cv.IMREAD_GRAYSCALE)
    pred_color = np.zeros((img.shape[0], img.shape[1], 3))
    pred_color[pred == 1] = road_color

    img[pred == 1] = alpha * pred_color[pred == 1] \
        + (1 - alpha) * img[pred == 1]
    out.write(img)

    # plt.close('all')
    # plt.cla()
    # plt.clf()
    # fig, axes = plt.subplots()
    # fig.set_size_inches(w * 2 // 100, h * 2 // 100)
    # plt.tight_layout()
    # axes[0].axis('off')
    # axes[0].imshow(img)
    # axes[0].imshow(pred_color, alpha=0.5)
    # plt.subplots_adjust(left=0, bottom=0)
    # fig.canvas.draw()

    # w, h = fig.canvas.get_width_height()
    # buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    # buf = buf.reshape((h, w, 4))
    # out_img = np.roll(buf, 3, axis=2)

out.release()
