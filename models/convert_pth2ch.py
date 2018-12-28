#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import serializers
import drn
from drn_pytorch import drn_c_26
from drn_pytorch import drn_d_105
import numpy as np
import torch
from torch.autograd import Variable


def save_model(chainer_model, pytorch_model, save_fname):
    np_x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    torch_x = Variable(torch.from_numpy(np_x))

    print('forward...')
    y = pytorch_model(torch_x)
    print('done')

    for name, param in pytorch_model.named_parameters():
        part_names = name.split('.')[:-1]
        module = getattr(pytorch_model, part_names[0])
        for part_name in part_names[1:]:
            module = getattr(module, part_name)

        link = getattr(chainer_model, part_names[0])
        for part_name in part_names[1:]:
            if hasattr(link, part_name):
                link = getattr(link, part_name)
            else:
                link = link[int(part_name)]

        if isinstance(module, torch.nn.Conv2d):
            if module.dilation[0] > 1 or module.dilation[1] > 1:
                link.dilate = module.dilation
            link.W.array = module.weight.data.numpy()
            if link.b is not None:
                link.b.array = module.bias.data.numpy()

        elif isinstance(module, torch.nn.BatchNorm2d):
            assert link.gamma.array.shape == module.weight.data.numpy().shape
            assert link.beta.array.shape == module.bias.data.numpy().shape
            link.decay = module.momentum
            link.eps = module.eps
            link.avg_mean = module.running_mean.numpy()
            link.avg_var = module.running_var.numpy()
            link.gamma.array = module.weight.data.numpy()
            link.beta.array = module.bias.data.numpy()

        else:
            raise ValueError(
                'Unknown module type: {} in {}'.format(type(module), name))

        print(name)

    n_params_t = len(list(pytorch_model.parameters()))
    n_params_c = len(list(chainer_model.params()))
    assert n_params_t == n_params_c, \
        'number of params differ: {} != {}'.format(n_params_t, n_params_c)

    serializers.save_npz(save_fname, chainer_model)

    ch_y = chainer_model(np_x)
    ch_y = ch_y.array

    print(ch_y.shape, y.data.numpy().shape)

    print(np.argmax(ch_y.ravel()))
    print(np.argmax(y.data.numpy().ravel()))

    print(np.testing.assert_array_almost_equal(
        ch_y, y.data.numpy(), decimal=4))


chainer_model = drn.drn_c_26()
pytorch_model = drn_c_26(pretrained=True)
save_model(chainer_model, pytorch_model, 'drn_c_26.npz')

chainer_model = drn.drn_d_105()
pytorch_model = drn_d_105(pretrained=True)
save_model(chainer_model, pytorch_model, 'drn_d_105.npz')
