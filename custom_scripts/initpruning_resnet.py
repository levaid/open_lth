import sys
sys.path.append('/home/levaid/bigstorage/open_lth')
from open_lth import *
from pruning import sparse_global
from models import base, initializers
from lottery.desc import LotteryDesc
from foundations import hparams
import torch
import torch.nn.functional as F
import torch.nn as nn
import datetime
from copy import deepcopy
from torch.nn.utils import prune
import torchvision
import time

import numpy as np
import os
from torchvision.datasets import CIFAR10


PRUNING_STRATEGY = sys.argv[1]
NETWORK_SIZE = int(sys.argv[2])

if PRUNING_STRATEGY not in ['snip', 'sparse']:
    raise(BaseException)

if NETWORK_SIZE not in [20, 38]:
    raise(BaseException)


LEVELS = 8
PRUNING_RATIO = 0.5
EPOCHS = 60
EXPERIMENT_NAME = datetime.datetime.now().strftime(
    "%Y_%m_%d_%H_%M_%S") + f'_{PRUNING_STRATEGY}_resnet{NETWORK_SIZE}'

print(EXPERIMENT_NAME)


class Model(base.Model):
    """A residual neural network as originally designed for CIFAR-10."""

    class Block(nn.Module):
        """A ResNet block."""

        def __init__(self, f_in: int, f_out: int, downsample=False):
            super(Model.Block, self).__init__()

            stride = 2 if downsample else 1
            self.conv1 = nn.Conv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(f_out)
            self.conv2 = nn.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(f_out)

            # No parameters for shortcut connections.
            if downsample or f_in != f_out:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(f_out)
                )
            else:
                self.shortcut = nn.Sequential()

        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return F.relu(out)

    def __init__(self, plan, initializer, outputs=None):
        super(Model, self).__init__()
        outputs = outputs or 10

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = nn.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(current_filters)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(Model.Block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        # Final fc layer. Size = number of filters in last segment.
        self.fc = nn.Linear(plan[-1][0], outputs)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize.
        self.apply(initializer)

        self.grads = {}

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('cifar_resnet_') and
                5 > len(model_name.split('_')) > 2 and
                all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[2:]]) and
                (int(model_name.split('_')[2]) - 2) % 6 == 0 and
                int(model_name.split('_')[2]) > 2)

    @staticmethod
    def get_model_from_name(model_name, initializer,  outputs=10):
        """The naming scheme for a ResNet is 'cifar_resnet_N[_W]'.

        The ResNet is structured as an initial convolutional layer followed by three "segments"
        and a linear output layer. Each segment consists of D blocks. Each block is two
        convolutional layers surrounded by a residual connection. Each layer in the first segment
        has W filters, each layer in the second segment has 32W filters, and each layer in the
        third segment has 64W filters.

        The name of a ResNet is 'cifar_resnet_N[_W]', where W is as described above.
        N is the total number of layers in the network: 2 + 6D.
        The default value of W is 16 if it isn't provided.

        For example, ResNet-20 has 20 layers. Exclusing the first convolutional layer and the final
        linear layer, there are 18 convolutional layers in the blocks. That means there are nine
        blocks, meaning there are three blocks per segment. Hence, D = 3.
        The name of the network would be 'cifar_resnet_20' or 'cifar_resnet_20_16'.
        """

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        name = model_name.split('_')
        W = 16 if len(name) == 3 else int(name[3])
        D = int(name[2])
        if (D - 2) % 3 != 0:
            raise ValueError('Invalid ResNet depth: {}'.format(D))
        D = (D - 2) // 6
        plan = [(W, D), (2*W, D), (4*W, D)]

        return Model(plan, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    
    def prune_network_sparse(self, pruning_ratio):
        weights_on_important_layers = []
        for name, module in self.named_modules():
            if 'conv' in name:
                # print(name)
                weights_on_important_layers += [
                    np.abs(module.weight.detach().cpu().numpy().flatten())]

        weights = np.concatenate(weights_on_important_layers)
        threshold = np.percentile(weights, (1-pruning_ratio)*100)
        for name, module in self.named_modules():
            if 'conv' in name:
                # print(name)
                prune.custom_from_mask(module, name='weight',
                                       mask=(torch.abs(module.weight) > torch.tensor(threshold, device=device)).to(device))

        return(threshold)

    def prune_network_snip(self, pruning_ratio):
        weights_on_important_layers = []
        for name, module in self.named_modules():
            if 'conv' in name:
                # print(name)
                weights_on_important_layers += [np.abs((module.weight.detach().cpu().numpy() *
                                                        self.grads[name+'.weight']).flatten())]

        weights = np.concatenate(weights_on_important_layers)
        threshold = np.percentile(weights, (1-pruning_ratio)*100)
        for name, module in self.named_modules():
            if 'conv' in name:
                # print(name)
                prune.custom_from_mask(module, name='weight',
                                       mask=(torch.abs(module.weight.detach().cpu() * self.grads[name+'.weight']) > torch.tensor(threshold, device=device)).to(device))

        return(threshold)


train_dataset = CIFAR10('/home/levaid/bigstorage/open_lth_datasets/cifar10', train=True,
                        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
test_dataset = CIFAR10('/home/levaid/bigstorage/open_lth_datasets/cifar10', train=False,
                       transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
original_model = Model.get_model_from_name(
    f'cifar_resnet_{NETWORK_SIZE}', initializers.kaiming_normal)


os.mkdir(os.path.join('/home/levaid/bigstorage/open_lth_data/', EXPERIMENT_NAME))


for level in range(LEVELS):
    performance_metrics = []
    model = deepcopy(original_model)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.1)

    current_ratio = PRUNING_RATIO ** (level + 1)

    if PRUNING_STRATEGY == 'sparse':
        threshold = model.prune_network_sparse(current_ratio)
        print(f'sparse pruning with {threshold:.6f} threshold')

    for ep in range(EPOCHS):
        starttime = time.time()
        model.train()

        for it, (examples, labels) in enumerate(train_loader):
            examples = examples.to(device=device)
            labels = labels.to(device=device)

            optimizer.zero_grad()

            loss = model.loss_criterion(model(examples), labels)

            for name, layer in model.named_parameters():
                if layer.requires_grad:
                    layer.retain_grad()

            loss.backward()

            for name, layer in model.named_parameters():
                if layer.requires_grad:
                    if name in model.grads:
                        model.grads[name] += np.abs(layer.grad.clone().cpu().numpy())
                    else:
                        model.grads[name] = np.abs(layer.grad.clone().cpu().numpy())
                    # print(layer.grad)

            optimizer.step()

        correct = 0
        model.eval()
        for it, (examples, labels) in enumerate(test_loader):
            examples = examples.to(device=device)
            labels = labels.to(device=device)

            correct += torch.sum(torch.argmax(model(examples),
                                              dim=1) == labels).cpu().numpy()
        performance_metrics += [(f'test_accuracy,{ep},{correct/10000.0}')]
        print(performance_metrics[-1] + f' time: {time.time()-starttime:.2f}s')

        if ep == 0 and PRUNING_STRATEGY == 'snip':
            threshold = model.prune_network_snip(current_ratio)
            print(f'snip pruning with {threshold:.6f} threshold')

        if ep == 40:
            print('reducing LR')
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10

        with open(f'/home/levaid/bigstorage/open_lth_data/{EXPERIMENT_NAME}/level_{level}_perf.log', 'w') as f:
            for line in performance_metrics:
                f.write(line + '\n')

    torch.save(model, f'/home/levaid/bigstorage/open_lth_data/{EXPERIMENT_NAME}/model_resnet_{NETWORK_SIZE}_level_{level}_.pth')

