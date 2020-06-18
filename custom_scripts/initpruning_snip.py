#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import pathlib
import json
from collections import defaultdict
import sys
sys.path.append('/home/levaid/bigstorage/open_lth')
from open_lth import *
import torchvision
from torch.nn.utils import prune
from copy import deepcopy
import datetime


# In[36]:


# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global
import initializers


class Model(base.Model):
    """A VGG-style neural network designed for CIFAR-10."""

    class ConvModule(nn.Module):
        """A single convolutional module in a VGG network."""

        def __init__(self, in_filters, out_filters):
            super(Model.ConvModule, self).__init__()
            # print([in_filters, out_filters])
            self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(out_filters)

        def forward(self, x):
            return F.relu(self.bn(self.conv(x)))

    def __init__(self, plan, initializer, outputs=10):
        super(Model, self).__init__()

        layers = []
        filters = 3
        print('plan:', plan)
        for spec in plan:
            if spec == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                # print([filters, spec])
                layers.append(Model.ConvModule(filters, spec))
                filters = spec

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, outputs)
        self.criterion = nn.CrossEntropyLoss()

        self.apply(initializer)

        self.grads = {}

    def forward(self, x):
        x = self.layers(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def prune_network_sparse(self, pruning_ratio):
        weights_on_important_layers = []
        for name, module in self.named_modules():
            if 'conv' in name or 'fc' in name:
                # print(name)
                weights_on_important_layers += [np.abs(module.weight.detach().cpu().numpy().flatten())]

        weights = np.concatenate(weights_on_important_layers)
        threshold = np.percentile(weights, (1-pruning_ratio)*100)
        for name, module in self.named_modules():
            if 'conv' in name or 'fc' in name:
                # print(name)
                prune.custom_from_mask(module, name = 'weight', 
                                       mask = (torch.abs(module.weight) > torch.tensor(threshold, device=device)).to(device))
                
        return(threshold)
    
    def prune_network_snip(self, pruning_ratio):
        weights_on_important_layers = []
        for name, module in self.named_modules():
            if 'conv' in name or 'fc' in name:
                # print(name)
                weights_on_important_layers += [np.abs((module.weight.detach().cpu().numpy() *
                                                        self.grads[name + '_orig']).flatten())]

        weights = np.concatenate(weights_on_important_layers)
        threshold = np.percentile(weights, (1-pruning_ratio)*100)
        for name, module in self.named_modules():
            if 'conv' in name or 'fc' in name:
                # print(name)
                prune.custom_from_mask(module, name = 'weight', 
                                       mask = (torch.abs(module.weight.detach().cpu() * self.grads[name + '_orig']) > torch.tensor(threshold, device=device)).to(device))
                
        return(threshold)

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('cifar_vgg_') and
                len(model_name.split('_')) == 3 and
                model_name.split('_')[2].isdigit() and
                int(model_name.split('_')[2]) in [11, 13, 16, 19])

    @staticmethod
    def get_model_from_name(model_name, initializer, outputs=10):
        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        outputs = outputs or 10

        num = int(model_name.split('_')[2])
        if num == 11:
            plan = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
        elif num == 13:
            plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
        elif num == 16:
            plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        elif num == 19:
            plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
        else:
            raise ValueError('Unknown VGG model: {}'.format(model_name))

        return Model(plan, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='cifar_vgg_16',
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
            pruning_layers_to_ignore='fc.weight'
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)


# In[5]:


from torchvision.datasets import CIFAR10

train_dataset = CIFAR10('/home/levaid/bigstorage/open_lth_datasets/cifar10', train=True, download=True,
                                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
test_dataset = CIFAR10('/home/levaid/bigstorage/open_lth_datasets/cifar10', train=False, download=True,
                                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)


# In[11]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
original_model = Model.get_model_from_name('cifar_vgg_11', initializers.kaiming_normal)


# In[12]:


LEVELS = 10
PRUNING_RATIO = 0.5
EPOCHS = 60

currentDT = datetime.datetime.now()

EXPERIMENT_NAME = currentDT.strftime("%Y_%m_%d_%H_%M_%S") + '_sparse_init'
print(EXPERIMENT_NAME)


# In[13]:


os.mkdir(os.path.join('/home/levaid/bigstorage/open_lth_data/', EXPERIMENT_NAME))


# In[37]:


performance_metrics = []
for level in range(LEVELS):
    
    model = deepcopy(original_model)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.1)

    current_ratio = PRUNING_RATIO ** (level)

    print(current_ratio)
    for ep in range(EPOCHS):
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
                        model.grads[name] += layer.grad.clone().cpu().numpy()
                    else:
                        model.grads[name] = layer.grad.clone().cpu().numpy()
                    # print(layer.grad)
            
            optimizer.step()
            
        if ep == 0:
            threshold = model.prune_network_snip(current_ratio)
            print(threshold)

        correct = 0
        model.eval()
        for it, (examples, labels) in enumerate(test_loader):
            examples = examples.to(device=device)
            labels = labels.to(device=device)
            
            correct += torch.sum(torch.argmax(model(examples), dim=1) == labels).cpu().numpy()
        performance_metrics += [(f'test_accuracy,{ep},{correct/10000.0}')]
        print(performance_metrics[-1])

        if ep == 40:
            print('reducing LR')
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        
        with open(f'/home/levaid/bigstorage/open_lth_data/{EXPERIMENT_NAME}/level_{level}_perf.log', 'w') as f:
            for line in performance_metrics:
                f.write(line + '\n')
        


# In[25]:


model.layers[2].conv.weight.requires_grad


# In[27]:


for name, module in model.named_modules():
    print(name)


# In[28]:


model.grads.keys()


# In[35]:


model.layers[0].conv


# In[ ]:




