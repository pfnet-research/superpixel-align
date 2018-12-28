import math

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np
from sequential import Sequential


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    if dilation > 1:
        conv = L.DilatedConvolution2D(
            in_planes, out_planes, ksize=3, stride=stride,
            pad=padding, nobias=True, dilate=dilation)
    else:
        conv = L.Convolution2D(
            in_planes, out_planes, ksize=3, stride=stride,
            pad=padding, nobias=True)
    return conv


class BasicBlock(chainer.Chain):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(BasicBlock, self).__init__()
        with self.init_scope():
            self.conv1 = conv3x3(inplanes, planes, stride,
                                 padding=dilation[0], dilation=dilation[0])
            self.bn1 = L.BatchNormalization(planes)
            self.conv2 = conv3x3(planes, planes,
                                 padding=dilation[1], dilation=dilation[1])
            self.bn2 = L.BatchNormalization(planes)
            self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def __call__(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        out = F.relu(out)

        return out


class Bottleneck(chainer.Chain):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=(1, 1), residual=True):
        super(Bottleneck, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                inplanes, planes, ksize=1, nobias=True)
            self.bn1 = L.BatchNormalization(planes)
            if dilation[1] > 1:
                self.conv2 = L.DilatedConvolution2D(
                    planes, planes, ksize=3, stride=stride, pad=dilation[1],
                    nobias=True, dilate=dilation[1])
            else:
                self.conv2 = L.Convolution2D(
                    planes, planes, ksize=3, stride=stride, pad=dilation[1],
                    nobias=True)
            self.bn2 = L.BatchNormalization(planes)
            self.conv3 = L.Convolution2D(
                planes, planes * 4, ksize=1, nobias=True)
            self.bn3 = L.BatchNormalization(planes * 4)
            self.downsample = downsample
        self.stride = stride

    def __call__(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


class DRN(chainer.Chain):

    def __init__(self, block, layers, num_classes=1000,
                 channels=(16, 32, 64, 128, 256, 512, 512, 512),
                 out_map=False, out_middle=False, arch='D'):
        super(DRN, self).__init__()
        self.inplanes = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch

        self.mean = np.asarray([0.485, 0.456, 0.406])
        self.std = np.asarray([0.229, 0.224, 0.225])

        with self.init_scope():
            if arch == 'C':
                self.conv1 = L.Convolution2D(
                    3, channels[0], ksize=7, stride=1, pad=3, nobias=True)
                self.bn1 = L.BatchNormalization(channels[0])

                self.layer1 = self._make_layer(
                    BasicBlock, channels[0], layers[0], stride=1)
                self.layer2 = self._make_layer(
                    BasicBlock, channels[1], layers[1], stride=2)
            elif arch == 'D':
                self.layer0 = Sequential(
                    L.Convolution2D(
                        3, channels[0], ksize=7, stride=1, pad=3, nobias=True),
                    L.BatchNormalization(channels[0]),
                    F.relu
                )

                self.layer1 = self._make_conv_layers(
                    channels[0], layers[0], stride=1)
                self.layer2 = self._make_conv_layers(
                    channels[1], layers[1], stride=2)

            self.layer3 = self._make_layer(
                block, channels[2], layers[2], stride=2)
            self.layer4 = self._make_layer(
                block, channels[3], layers[3], stride=2)
            self.layer5 = self._make_layer(
                block, channels[4], layers[4], dilation=2, new_level=False)
            self.layer6 = None if layers[5] == 0 else \
                self._make_layer(block, channels[5], layers[5], dilation=4,
                                 new_level=False)

            if arch == 'C':
                self.layer7 = None if layers[6] == 0 else \
                    self._make_layer(
                        BasicBlock, channels[6], layers[6], dilation=2,
                        new_level=False, residual=False)
                self.layer8 = None if layers[7] == 0 else \
                    self._make_layer(
                        BasicBlock, channels[7], layers[7], dilation=1,
                        new_level=False, residual=False)
            elif arch == 'D':
                self.layer7 = None if layers[6] == 0 else \
                    self._make_conv_layers(channels[6], layers[6], dilation=2)
                self.layer8 = None if layers[7] == 0 else \
                    self._make_conv_layers(channels[7], layers[7], dilation=1)

            if num_classes > 0:
                self.fc = L.Convolution2D(
                    self.out_dim, num_classes, ksize=1, stride=1, pad=0,
                    nobias=False)
            for m in self.children():
                if isinstance(m, L.Convolution2D):
                    if type(m.ksize) == int:
                        ksize = (m.ksize, m.ksize)
                    n = ksize[0] * ksize[1] * m.out_channels
                    m.W.array = np.random.randn(*m.W.shape) * math.sqrt(2. / n)
                elif isinstance(m, L.BatchNormalization):
                    m.gamma.array.fill(1)
                    m.beta.array.fill(0)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
                    new_level=True, residual=True):
        assert dilation == 1 or dilation % 2 == 0
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                L.Convolution2D(self.inplanes, planes * block.expansion,
                                ksize=1, stride=stride, nobias=True),
                L.BatchNormalization(planes * block.expansion),
            )

        layers = list()
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, residual=residual,
                                dilation=(dilation, dilation)))

        return Sequential(*layers)

    def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            if dilation > 1:
                conv = L.DilatedConvolution2D(
                    self.inplanes, channels, ksize=3,
                    stride=stride if i == 0 else 1,
                    pad=dilation, nobias=True, dilate=dilation)
            else:
                conv = L.Convolution2D(
                    self.inplanes, channels, ksize=3,
                    stride=stride if i == 0 else 1, pad=dilation, nobias=True)
            modules.extend([
                conv,
                L.BatchNormalization(channels),
                F.relu
            ])
            self.inplanes = channels
        return Sequential(*modules)

    def __call__(self, x):
        if self.out_middle:
            y = list()

        if self.arch == 'C':
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
        elif self.arch == 'D':
            x = self.layer0(x)

        x = self.layer1(x)
        if self.out_middle:
            y.append(x)
        x = self.layer2(x)
        if self.out_middle:
            y.append(x)

        x = self.layer3(x)
        if self.out_middle:
            y.append(x)

        x = self.layer4(x)
        if self.out_middle:
            y.append(x)

        x = self.layer5(x)
        if self.out_middle:
            y.append(x)

        if self.layer6 is not None:
            x = self.layer6(x)
            if self.out_middle:
                y.append(x)

        if self.layer7 is not None:
            x = self.layer7(x)
            if self.out_middle:
                y.append(x)

        if self.layer8 is not None:
            x = self.layer8(x)
            if self.out_middle:
                y.append(x)

        if self.out_map:
            x = self.fc(x)
        else:
            x = F.average_pooling_2d(x, x.shape[2:])
            x = self.fc(x)
            x = x.reshape(x.shape[0], -1)

        if self.out_middle:
            return x, y
        else:
            return x

    def predict(self, x):
        """Prediction method

        Args:
            x (ndarray): The value range should be [0, 255] and RGB order.

        """
        if isinstance(x, cuda.cupy.ndarray):
            mean = cuda.cupy.asarray(self.mean)
            std = cuda.cupy.asarray(self.std)
        else:
            mean, std = self.mean, self.std
        x /= 255.
        x -= mean[:, None, None]
        x /= std[:, None, None]
        return self(x[None, :, :])

    def batch_predict(self, x):
        """Prediction method

        Args:
            x (ndarray): The value range should be [0, 255] and RGB order.
                The shape should be (N, C, H, W)

        """
        assert x.ndim == 4
        if isinstance(x, cuda.cupy.ndarray):
            mean = cuda.cupy.asarray(self.mean)
            std = cuda.cupy.asarray(self.std)
        else:
            mean, std = self.mean, self.std

        x /= 255.
        x -= mean[None, :, None, None]
        x /= std[None, :, None, None]
        with chainer.using_config('train', False), \
                chainer.function.no_backprop_mode():
            y = self(chainer.Variable(x))
        return y


def drn_c_26(pretrained=False, **kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='C', **kwargs)
    return model


def drn_d_105(**kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 1, 1], arch='D', **kwargs)
    return model
