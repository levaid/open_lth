# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
import numpy as np
import torch.optim as optim
import math

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global


class Model(base.Model):
    """A VGG-style neural network designed for CIFAR-10."""

    class Fire(nn.Module):
        def __init__(self, inplanes, squeeze_planes, expand_planes):
            super(Model.Fire, self).__init__()
            self.conv1 = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=1)
            self.bn1 = nn.BatchNorm2d(squeeze_planes)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=1, stride=1)
            self.bn2 = nn.BatchNorm2d(expand_planes)
            self.conv3 = nn.Conv2d(squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(expand_planes)
            self.relu2 = nn.ReLU(inplace=True)

            # using MSR initilization
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    m.weight.data.normal_(0, math.sqrt(2./n))

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            out1 = self.conv2(x)
            out1 = self.bn2(out1)
            out2 = self.conv3(x)
            out2 = self.bn3(out2)
            out = torch.cat([out1, out2], 1)
            out = self.relu2(out)
            return out


    def __init__(self, initializer, outputs):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1) # 32
        self.bn1 = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 16
        self.fire2 = Model.Fire(96, 16, 64)
        self.fire3 = Model.Fire(128, 16, 64)
        self.fire4 = Model.Fire(128, 32, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 8
        self.fire5 = Model.Fire(256, 32, 128)
        self.fire6 = Model.Fire(256, 48, 192)
        self.fire7 = Model.Fire(384, 48, 192)
        self.fire8 = Model.Fire(384, 64, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 4
        self.fire9 = Model.Fire(512, 64, 256)
        self.conv2 = nn.Conv2d(512, outputs, kernel_size=1, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.softmax = nn.Softmax(dim=1)
        
        for m in self.modules():
            # HACK
            break
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


        self.criterion = nn.CrossEntropyLoss()

        self.apply(initializer)

        self.grads = {}

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.maxpool2(x)
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.maxpool3(x)
        x = self.fire9(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.softmax(x)
        x = torch.squeeze(x)
        return x

    @property
    def output_layer_names(self):
        return ['conv2.weight', 'conv2.bias']

    def is_valid_model_name(model_name: str) -> bool:
        return model_name == 'squeezenet_cifar'

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs):
        if model_name != 'squeezenet_cifar':
            raise ValueError('Invalid model name: {}'.format(model_name))
        
        assert outputs is not None
        outputs = outputs

        return Model(initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='squeezenet_cifar',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='cifar10',
            batch_size=128
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            momentum=0.9,
            milestone_steps='80ep,120ep',
            lr=0.1,
            gamma=0.1,
            weight_decay=1e-4,
            training_steps='160ep'
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore='conv2.weight'
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
