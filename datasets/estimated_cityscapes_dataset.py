import glob
import os

from chainer import dataset
from chainercv import transforms
from chainercv import utils
import numpy as np


class EstimatedCityscapesDataset(dataset.DatasetMixin):

    def __init__(self, img_dir, label_dir, resize_shape, random=False,
                 use_soft_label=False):
        if not os.path.exists(img_dir):
            raise ValueError('{} does not exist'.format(img_dir))
        if not os.path.exists(label_dir):
            raise ValueError('{} does not exist'.format(label_dir))

        endswith = '*leftImg8bit_scores.npy' \
            if use_soft_label else '*leftImg8bit.npy'
        self.label_paths = [
            fn for fn in glob.glob(os.path.join(label_dir, endswith))]

        self.img_paths = list()
        for label_path in self.label_paths:
            city_name = os.path.basename(label_path).split('_')[0]
            if use_soft_label:
                label_path = label_path.replace('_scores', '')
            label_fnbase = os.path.splitext(os.path.basename(label_path))[0]
            img_path = os.path.join(img_dir, city_name, label_fnbase + '.png')
            self.img_paths.append(img_path)

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

    def __len__(self):
        return len(self.img_paths)

    def get_example(self, i):
        img = utils.read_image(self.img_paths[i])
        label = np.load(self.label_paths[i])
        if not self.use_soft_label:
            label = label.astype(np.int32)
        else:
            label = label.astype(np.float32)

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
