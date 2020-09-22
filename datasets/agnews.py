from datasets.base import Dataset
import os
from PIL import Image

from datasets import base
from platforms.platform import get_platform


from torch.utils.data import Dataset as TorchDataset
import torch
import json
import csv

class AGNews(TorchDataset):
    
    def __len__(self):
        return len(self.targets)


    def __getitem__(self, idx):
        X = self.oneHotEncode(idx)
        y = self.y[idx]
        return X, y

    def loadAlphabet(self, alphabet_path):
        with open(alphabet_path) as f:
            self.alphabet = ''.join(json.load(f))

    def load(self, label_data_path, lowercase = True):
        self.targets = []
        self.data = []
        with open(label_data_path, 'r') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            # num_samples = sum(1 for row in rdr)
            for index, row in enumerate(rdr):
                self.targets.append(int(row[0])-1)
                txt = ' '.join(row[1:])
                if lowercase:
                    txt = txt.lower()                
                self.data.append(txt)

        self.y = torch.LongTensor(self.targets)

    def __init__(self, train, l0 = 1014):
        """Create AG's News dataset object.
        Arguments:
            l0: max length of a sample.
        """
        label_data_path = os.path.join(get_platform().dataset_root, 'agnews', f'{"train" if train else "test"}.txt')
        self.l0 = l0

        alphabet_path = os.path.join(get_platform().dataset_root, 'agnews', 'alphabet.json')
        # label_data_path = os.path.join(get_platform().dataset_root, 'agnews', 'train.csv')
        # read alphabet
        self.loadAlphabet(alphabet_path)
        self.load(label_data_path)


    def oneHotEncode(self, idx):
        # X = (batch, 70, sequence_length)
        X = torch.zeros(len(self.alphabet), self.l0)
        sequence = self.data[idx]
        for index_char, char in enumerate(sequence[::-1]):
            if self.char2Index(char) != -1:
                X[self.char2Index(char)][index_char] = 1.0
        return X

    def char2Index(self, character):
        return self.alphabet.find(character)

    def getClassWeight(self):
        num_samples = self.__len__()
        label_set = set(self.label)
        num_class = [self.label.count(c) for c in label_set]
        class_weight = [num_samples/float(self.label.count(c)) for c in label_set]    
        return class_weight, num_class

class Dataset(AGNews):

    @staticmethod
    def num_train_examples(): return 120000

    @staticmethod
    def num_test_examples(): return 7200

    @staticmethod
    def num_classes(): return 4

    @staticmethod
    def get_train_set(use_augmentation=False): # augmentation has no use in NLP
        train_set = AGNews(train=True)
        # train_loader = DataLoader(train_set, batch_size=32, num_workers=1)
        return train_set

    @staticmethod
    def get_test_set():
        test_set = AGNews(train=False)
        return test_set


DataLoader = base.TextLoader