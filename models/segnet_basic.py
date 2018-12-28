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


class SegNetBasic(chainer.Chain):

    def __init__(self, n_class=None, comm=None, pred_shape=None):
        super(SegNetBasic, self).__init__()
        w = chainer.initializers.HeNormal()
        if comm is not None:
            bn = partial(MultiNodeBatchNormalization, comm=comm)
        else:
            bn = L.BatchNormalization
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 7, 1, 3, True, w)
            self.conv1_bn = bn(64, initial_beta=0.001)
            self.conv2 = L.Convolution2D(64, 64, 7, 1, 3, True, w)
            self.conv2_bn = bn(64, initial_beta=0.001)
            self.conv3 = L.Convolution2D(64, 64, 7, 1, 3, True, w)
            self.conv3_bn = bn(64, initial_beta=0.001)
            self.conv4 = L.Convolution2D(64, 64, 7, 1, 3, True, w)
            self.conv4_bn = bn(64, initial_beta=0.001)
            self.conv_decode4 = L.Convolution2D(64, 64, 7, 1, 3, True, w)
            self.conv_decode4_bn = bn(64, initial_beta=0.001)
            self.conv_decode3 = L.Convolution2D(64, 64, 7, 1, 3, True, w)
            self.conv_decode3_bn = bn(64, initial_beta=0.001)
            self.conv_decode2 = L.Convolution2D(64, 64, 7, 1, 3, True, w)
            self.conv_decode2_bn = bn(64, initial_beta=0.001)
            self.conv_decode1 = L.Convolution2D(64, 64, 7, 1, 3, True, w)
            self.conv_decode1_bn = bn(64, initial_beta=0.001)
            self.conv_classifier = L.Convolution2D(
                64, n_class, 1, 1, 0, False, w)

        self.n_class = n_class
        self.pred_shape = pred_shape

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
        h = F.local_response_normalization(x, 5, 1, 1e-4 / 5., 0.75)
        h = _without_cudnn(p1, F.relu(self.conv1_bn(self.conv1(h))))
        h = _without_cudnn(p2, F.relu(self.conv2_bn(self.conv2(h))))
        h = _without_cudnn(p3, F.relu(self.conv3_bn(self.conv3(h))))
        h = _without_cudnn(p4, F.relu(self.conv4_bn(self.conv4(h))))
        h = self._upsampling_2d(h, p4)
        h = self.conv_decode4_bn(self.conv_decode4(h))
        h = self._upsampling_2d(h, p3)
        h = self.conv_decode3_bn(self.conv_decode3(h))
        h = self._upsampling_2d(h, p2)
        h = self.conv_decode2_bn(self.conv_decode2(h))
        h = self._upsampling_2d(h, p1)
        h = self.conv_decode1_bn(self.conv_decode1(h))
        score = self.conv_classifier(h)
        return score

    def predict(self, imgs, return_score=False):
        """Conduct semantic segmentations from images.

        Args:
            imgs (iterable of numpy.ndarray): Arrays holding images.
                All images are in CHW and RGB format
                and the range of their values are :math:`[0, 255]`.
            return_score (bool): If True, this function will also return the
                softmax result of predicted label with the argmax result.
        Returns:
            list of numpy.ndarray:
            List of integer labels predicted from each image in the input \
            list.

        """
        labels = list()
        for img in imgs:
            with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                x = chainer.Variable(self.xp.asarray(img[np.newaxis]))
                y = self.__call__(x)
                if return_score:
                    y = F.softmax(y)
                score = y[0].data
            score = chainer.cuda.to_cpu(score)
            if self.pred_shape is not None:
                if score.shape[1:] != self.pred_shape:
                    dtype = score.dtype
                    score = transforms.resize(
                        score, self.pred_shape).astype(dtype)
            label = np.argmax(score, axis=0).astype(np.int32)
            if return_score:
                labels.append((label, score))
            else:
                labels.append(label)
        return labels
