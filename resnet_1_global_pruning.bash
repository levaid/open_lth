#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python open_lth.py lottery --model_name=cifar_resnet_38_64 --levels=6 --dataset_name=cifar10 --batch_size=128  --model_init=kaiming_normal --batchnorm_init=uniform --optimizer_name=sgd --lr=0.1 --momentum=0.1 --training_steps=40ep --pruning_strategy=sparse_global --pruning_fraction=0.5 --replicate=3
