from functools import partial

import chainer
import chainer.functions as F
import chainer.links as L
from chainercv import transforms
from chainermn.links import MultiNodeBatchNormalization
import numpy as np


def _without_cudnn(f, x):
    with chainer.using_config('use_cudnn', 'never'):
        return f.apply((x,))[0]


class CBR(chainer.Chain):

    def __init__(self, n_ch, comm=None):
        super().__init__()
        if comm is not None:
            bn = partial(MultiNodeBatchNormalization, comm=comm)
        else:
            bn = L.BatchNormalization
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.conv = L.Convolution2D(None, n_ch, 3, 1, 1, True, w)
            self.bn = bn(n_ch)

    def __call__(self, x):
        return F.relu(self.bn(self.conv(x)))


class Block(chainer.ChainList):

    def __init__(self, n_cbr, mid_ch, out_ch, comm=None):
        super().__init__()
        for _ in range(n_cbr - 1):
            self.add_link(CBR(mid_ch, comm))
        self.add_link(CBR(out_ch, comm))

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class SegNet(chainer.Chain):

    def __init__(self, n_class, comm=None):
        super().__init__()
        w = chainer.initializers.HeNormal()
        with self.init_scope():
            self.block1 = Block(2, 64, 64, comm)
            self.block2 = Block(2, 128, 128, comm)
            self.block3 = Block(3, 256, 256, comm)
            self.block4 = Block(3, 512, 512, comm)
            self.block5 = Block(3, 512, 512, comm)
            self.up_block5 = Block(3, 512, 512, comm)
            self.up_block4 = Block(3, 512, 256, comm)
            self.up_block3 = Block(3, 256, 128, comm)
            self.up_block2 = Block(2, 128, 64, comm)
            self.up_block1 = CBR(64, comm)
            self.score = L.Convolution2D(None, n_class, 3, 1, 1, False, w)
        self.n_class = n_class

    def _upsampling_2d(self, x, pool):
        if x.shape != pool.indexes.shape:
            min_h = min(x.shape[2], pool.indexes.shape[2])
            min_w = min(x.shape[3], pool.indexes.shape[3])
            x = x[:, :, :min_h, :min_w]
            pool.indexes = pool.indexes[:, :, :min_h, :min_w]
        outsize = (x.shape[2] * 2, x.shape[3] * 2)
        return F.upsampling_2d(
            x, pool.indexes, ksize=(pool.kh, pool.kw),
            stride=(pool.sy, pool.sx), pad=(pool.ph, pool.pw), outsize=outsize)

    def __call__(self, x):
        p1 = F.MaxPooling2D(2, 2)
        p2 = F.MaxPooling2D(2, 2)
        p3 = F.MaxPooling2D(2, 2)
        p4 = F.MaxPooling2D(2, 2)
        p5 = F.MaxPooling2D(2, 2)

        h = _without_cudnn(p1, self.block1(x))
        h = _without_cudnn(p2, self.block2(h))
        h = _without_cudnn(p3, self.block3(h))
        h = _without_cudnn(p4, self.block4(h))
        h = _without_cudnn(p5, self.block5(h))

        h = self.up_block5(self._upsampling_2d(h, p5))
        h = self.up_block4(self._upsampling_2d(h, p4))
        h = self.up_block3(self._upsampling_2d(h, p3))
        h = self.up_block2(self._upsampling_2d(h, p2))
        h = self.up_block1(self._upsampling_2d(h, p1))
        return self.score(h)

    def predict(self, imgs):
        """Conduct semantic segmentations from images.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their values are :math:`[0, 255]`.
        Returns:
            list of numpy.ndarray:
            List of integer labels predicted from each image in the input \
            list.

        """
        labels = list()
        for img in imgs:
            C, H, W = img.shape
            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                x = chainer.Variable(self.xp.asarray(img[np.newaxis]))
                score = self.__call__(x)[0].data
            score = chainer.cuda.to_cpu(score)
            if score.shape != (C, H, W):
                dtype = score.dtype
                score = transforms.resize(score, (H, W)).astype(dtype)

            label = np.argmax(score, axis=0).astype(np.int32)
            labels.append(label)
        return labels
