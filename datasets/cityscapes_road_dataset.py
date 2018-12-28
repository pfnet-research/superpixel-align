import glob
import os

from chainer import dataset
from chainercv import transforms
from chainercv import utils
import numpy as np


class CityscapesRoadDataset(dataset.DatasetMixin):

    def __init__(self, data_dir, resize_shape, resol='gtFine', split='val'):
        if not os.path.exists(data_dir):
            raise ValueError('{} does not exist'.format(data_dir))
        self.label_fns = sorted(glob.glob(
            os.path.join(data_dir, resol, split, '*/*labelIds.png')))

        img_dir = os.path.join(data_dir, 'leftImg8bit', split)
        self.img_fns = []
        for label_fn in self.label_fns:
            city, seq, frame = os.path.basename(label_fn).split('_')[:3]
            img_fn_base = '_'.join([city, seq, frame]) + '_leftImg8bit.png'
            img_fn = os.path.join(img_dir, city, img_fn_base)
            self.img_fns.append(img_fn)

        assert len(self.label_fns) == len(self.img_fns)

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

    def __len__(self):
        return len(self.label_fns)

    def get_example(self, i):
        img = utils.read_image(self.img_fns[i])
        label = utils.read_image(self.label_fns[i], np.int32, color=False)[0]

        out_label = np.zeros(label.shape, dtype=np.int32)
        for i in self.ignore_ids:
            out_label[label == i] = -1
        for i in self.road_ids:
            out_label[label == i] = 1

        if img.shape[1] != self.resize_shape[0] \
                or img.shape[2] != self.resize_shape[1]:
            img = transforms.resize(img, self.resize_shape, 3)
            # out_label = transforms.resize(
            #     out_label[None, ...], self.resize_shape, 0)[0]

        img -= self.mean[:, None, None]
        img /= self.std[:, None, None]

        return img, out_label
