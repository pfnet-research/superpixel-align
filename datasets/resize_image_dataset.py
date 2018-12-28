from PIL import Image
from chainer.dataset import dataset_mixin
from chainercv import transforms
import numpy as np


class ResizeImageDataset(dataset_mixin.DatasetMixin):

    def __init__(self, paths, resize_shape=None, dtype=np.float32):
        self._paths = paths
        self._resize_shape = resize_shape
        self._dtype = dtype

    def __len__(self):
        return len(self._paths)

    def get_example(self, i):
        path = self._paths[i]
        f = Image.open(path)
        try:
            image = np.asarray(f, dtype=np.uint8)
        finally:
            f.close()

        if image.ndim == 2:
            # image is greyscale
            image = image[:, :, np.newaxis]
        if image.shape[2] > 3:
            image = image[:, :, :3]

        image = image.transpose(2, 0, 1)

        if self._resize_shape is not None:
            image = transforms.resize(image, self._resize_shape, 3)

        return image.astype(self._dtype)
