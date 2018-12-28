import os
import zipfile

from chainer import dataset
from chainercv import utils
import cv2 as cv
import numpy as np


class ZippedCityscapesRoadDataset(dataset.DatasetMixin):

    def __init__(self, img_zip_fn, label_zip_fn, resize_shape,
                 standardize=True):
        if not os.path.exists(img_zip_fn):
            raise ValueError('{} does not exist'.format(img_zip_fn))
        if not os.path.exists(label_zip_fn):
            raise ValueError('{} does not exist'.format(label_zip_fn))

        label_zf = zipfile.ZipFile(label_zip_fn)
        label_fns = {
            '_'.join(os.path.basename(fn).split('_')[:3]): fn
            for fn in label_zf.namelist() if fn.endswith('labelIds.png')}

        img_zf = zipfile.ZipFile(img_zip_fn)
        img_fns = {
            '_'.join(os.path.basename(fn).split('_')[:3]): fn
            for fn in img_zf.namelist() if fn.endswith('leftImg8bit.png')}

        keys = img_fns.keys() \
            if len(img_fns) < len(label_fns) else label_fns.keys()

        self.img_fns = []
        self.label_fns = []
        for key in keys:
            self.img_fns.append(img_fns[key])
            self.label_fns.append(label_fns[key])

        if standardize:
            self.mean = np.array([
                7.315835921071366954e+01,
                8.290891754262415247e+01,
                7.239239876194160672e+01], dtype=np.float32)
            self.std = np.array([
                4.161211675686322309e+01,
                4.221582767516605372e+01,
                4.048309952494058450e+01], dtype=np.float32)

        self.resize_shape = resize_shape
        self.ignore_ids = [0, 1, 2, 3, 4, 5, 6]
        self.road_ids = [7]
        self.standardize = standardize
        self.img_zip_fn = img_zip_fn
        self.label_zip_fn = label_zip_fn
        self.label_zf = None
        self.img_zf = None

    def __len__(self):
        return len(self.label_fns)

    def get_example(self, i):
        if self.img_zf is None:
            self.img_zf = zipfile.ZipFile(self.img_zip_fn)
        if self.label_zf is None:
            self.label_zf = zipfile.ZipFile(self.label_zip_fn)

        img = utils.read_image(
            self.img_zf.open(self.img_fns[i]), np.uint8)
        label = utils.read_image(
            self.label_zf.open(self.label_fns[i]), np.int32, color=False)[0]

        out_label = np.zeros(label.shape, dtype=np.int32)
        for i in self.ignore_ids:
            out_label[label == i] = -1
        for i in self.road_ids:
            out_label[label == i] = 1

        # Resize only image
        if img.shape[1] != self.resize_shape[0] \
                or img.shape[2] != self.resize_shape[1]:
            h, w = self.resize_shape
            img = cv.resize(
                img.transpose(1, 2, 0), (w, h), interpolation=cv.INTER_CUBIC)
            img = img.transpose(2, 0, 1)
        img = img.astype(np.float32)

        if self.standardize:
            img -= self.mean[:, None, None]
            img /= self.std[:, None, None]

        return img, out_label
