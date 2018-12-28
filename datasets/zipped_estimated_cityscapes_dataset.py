import glob
import os
import zipfile

from chainer import dataset
from chainercv import transforms
from chainercv import utils
import numpy as np


class ZippedEstimatedCityscapesDataset(dataset.DatasetMixin):

    def __init__(self, img_zip_fn, label_zip_fn, resize_shape, random=False,
                 use_soft_label=False):
        if not os.path.exists(img_zip_fn):
            raise ValueError('{} does not exist'.format(img_zip_fn))
        if not os.path.exists(label_zip_fn):
            raise ValueError('{} does not exist'.format(label_zip_fn))

        label_zf = zipfile.ZipFile(label_zip_fn)
        postfix = 'leftImg8bit' + ('_scores.npy' if use_soft_label else '.npy')
        label_fns = {
            '_'.join(os.path.basename(fn).split('_')[:3]): fn
            for fn in label_zf.namelist() if fn.endswith(postfix)}

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

        self.mean = np.array([
            7.315835921071366954e+01,
            8.290891754262415247e+01,
            7.239239876194160672e+01], dtype=np.float32)
        self.std = np.array([
            4.161211675686322309e+01,
            4.221582767516605372e+01,
            4.048309952494058450e+01], dtype=np.float32)

        self.resize_shape = resize_shape
        self.random = random
        self.use_soft_label = use_soft_label
        self.img_zip_fn = img_zip_fn
        self.label_zip_fn = label_zip_fn
        self.label_zf = None
        self.img_zf = None

    def __len__(self):
        return len(self.img_fns)

    def get_example(self, i):
        if self.img_zf is None:
            self.img_zf = zipfile.ZipFile(self.img_zip_fn)
        if self.label_zf is None:
            self.label_zf = np.load(self.label_zip_fn)

        img = utils.read_image(self.img_zf.open(self.img_fns[i]))
        label = self.label_zf[self.label_fns[i]]
        if self.use_soft_label:
            label = label.astype(np.float32)
        else:
            label = label.astype(np.int32)

        if img.shape[1] != self.resize_shape[0] \
                or img.shape[2] != self.resize_shape[1]:
            img = transforms.resize(img, self.resize_shape, 3)
        if label.shape[1] != self.resize_shape[0] \
                or label.shape[2] != self.resize_shape[1]:
            if not self.use_soft_label:
                label = label[None, ...]
            label = transforms.resize(label, self.resize_shape, 0)
            if not self.use_soft_label:
                label = label[0]

        if self.random:
            img = transforms.pca_lighting(img, 25.5)
            if np.random.rand() > 0.5:
                img = img[:, :, ::-1]
                if self.use_soft_label:
                    label = label[:, :, ::-1]
                else:
                    label = label[:, ::-1]

        img -= self.mean[:, None, None]
        img /= self.std[:, None, None]

        return img, label
