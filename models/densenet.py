# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp


from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global
from collections import OrderedDict


def store(grad, name, grads):
    print(name)
    grads[name] = grad.clone()


class Model(base.Model):
    r"""Densenet-BC model class, based on
        `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
        Args:
            growth_rate (int) - how many filters to add each layer (`k` in paper)
            block_config (list of 3 or 4 ints) - how many layers in each pooling block
            num_init_features (int) - the number of filters to learn in the first convolution layer
            bn_size (int) - multiplicative factor for number of bottle neck layers
                (i.e. bn_size * k features in the bottleneck layer)
            drop_rate (float) - dropout rate after each dense layer
            num_classes (int) - number of classification classes
            small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
            efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
        """

    def _bn_function_factory(norm, relu, conv):
        def bn_function(*inputs):
            concated_features = torch.cat(inputs, 1)
            bottleneck_output = conv(relu(norm(concated_features)))
            return bottleneck_output

        return bn_function


    class _DenseLayer(nn.Module):
        def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
            super(Model._DenseLayer, self).__init__()
            self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
            self.add_module('relu1', nn.ReLU(inplace=True)),
            self.add_module('conv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                            kernel_size=1, stride=1, bias=False)),
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
            self.add_module('relu2', nn.ReLU(inplace=True)),
            self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                            kernel_size=3, stride=1, padding=1, bias=False)),
            self.drop_rate = drop_rate
            self.efficient = efficient

        def forward(self, *prev_features):
            bn_function = Model._bn_function_factory(self.norm1, self.relu1, self.conv1)
            if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
                bottleneck_output = cp.checkpoint(bn_function, *prev_features)
            else:
                bottleneck_output = bn_function(*prev_features)
            new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
            if self.drop_rate > 0:
                new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
            return new_features


    class _Transition(nn.Sequential):
        def __init__(self, num_input_features, num_output_features):
            super(Model._Transition, self).__init__()
            self.add_module('norm', nn.BatchNorm2d(num_input_features))
            self.add_module('relu', nn.ReLU(inplace=True))
            self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                            kernel_size=1, stride=1, bias=False))
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


    class _DenseBlock(nn.Module):
        def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
            super(Model._DenseBlock, self).__init__()
            for i in range(num_layers):
                layer = Model._DenseLayer(
                    num_input_features + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    drop_rate=drop_rate,
                    efficient=efficient,
                )
                self.add_module('denselayer%d' % (i + 1), layer)

        def forward(self, init_features):
            features = [init_features]
            for name, layer in self.named_children():
                new_features = layer(*features)
                features.append(new_features)
            return torch.cat(features, 1)


    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                num_init_features=24, bn_size=4, drop_rate=0,
                num_classes=10, small_inputs=True, efficient=False):

        super(Model, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

        # First convolution
        if small_inputs:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ]))
            self.features.add_module('norm0', nn.BatchNorm2d(num_init_features))
            self.features.add_module('relu0', nn.ReLU(inplace=True))
            self.features.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                        ceil_mode=False))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = Model._DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Model._Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        self.grads = {}
        self.loss_criterion = nn.CrossEntropyLoss()

        # Initialization
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = param.size(0) * param.size(2) * param.size(3)
                param.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                param.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                param.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    @property
    def output_layer_names(self):
        return ['classifier.weight', 'classifier.bias']

    @staticmethod
    def is_valid_model_name(model_name): #densenet_121_cifar
        return (model_name.startswith('densenet_') and
                len(model_name.split('_')) == 3 and 
                int(model_name.split('_')[1]) in {121, 161, 169, 201})

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs):
        """The name of a model is densenet_N.

        """
        configuration = int(model_name.split('_')[1])

        params_dict = {121: {'block_config': [6, 12, 24, 16], 'growth_rate': 32, 'num_init_features': 64}, 
                       161: {'block_config': [6, 12, 36, 24], 'growth_rate': 48, 'num_init_features': 96},
                       169: {'block_config': [6, 12, 32, 32], 'growth_rate': 32, 'num_init_features': 64},
                       201: {'block_config': [6, 12, 48, 32], 'growth_rate': 32, 'num_init_features': 64}}

        params_dict['num_classes'] = outputs


        return Model(**params_dict[configuration])

    @property
    def loss_criterion(self):
        return self.criterion

    

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='densenet_121_cifar',
            model_init='kaiming_normal',
            batchnorm_init='uniform'
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='cifar10',
            batch_size=128,
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            momentum=0.9,
            milestone_steps='80ep,120ep',
            lr=0.1,
            gamma=0.1,
            weight_decay=1e-4,
            training_steps='160ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)
