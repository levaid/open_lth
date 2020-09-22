import torch.nn as nn

import torch.nn as nn

from foundations import hparams
from lottery.desc import LotteryDesc
from models import base
from pruning import sparse_global

import torch
import torch.nn as nn
import torch.nn.functional as F

class  Model(base.Model):
    """Character-level CNN designed for text classification."""

    def __init__(self, num_classes=4):
        super(Model, self).__init__()

        # I do not support setting hparams since this model is very different from image classifiers.
        # AGNews is 70 dim on input, it has 70 different characters
        hparams = {'num_features': 70, 'dropout': 0.5, }

        self.conv1 = nn.Sequential(
            nn.Conv1d(hparams['num_features'], 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )            
            
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()    
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
            
        
        self.fc1 = nn.Sequential(
            nn.Linear(8704, 1024),
            nn.ReLU(),
            nn.Dropout(p=hparams['dropout'])
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=hparams['dropout'])
        )

        self.fc3 = nn.Linear(1024, num_classes)

        self.grads = {}
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # collapse
        x = x.view(x.size(0), -1)
        # linear layer
        x = self.fc1(x)
        # linear layer
        x = self.fc2(x)
        # linear layer
        x = self.fc3(x)
        
        return x

    @property
    def output_layer_names(self):
        return ['fc3.weight', 'fc3.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return model_name=='charcnn'

    @staticmethod
    def get_model_from_name(model_name, initializer,  outputs=4):
        """Name is charcnn."""

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))
            
        return(Model(num_classes=outputs))

    @property
    def loss_criterion(self):
        return self.criterion

    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='charcnn',
            model_init='kaiming_normal',
            batchnorm_init='uniform',
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='agnews',
            batch_size=32,
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            momentum=0.9,
            milestone_steps='120ep,180ep',
            lr=0.1,
            gamma=0.5,
            weight_decay=0,
            training_steps='200ep',
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams)




