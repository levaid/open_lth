#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python open_lth.py lottery --model_name=cifar_vgg_16 --levels=4 --rewinding_steps=3ep --transformation_seed=50 --dataset_name=cifar10 --batch_size=64  --model_init=kaiming_normal --batchnorm_init=uniform --optimizer_name=sgd --lr=0.01 --training_steps=10ep --pruning_strategy=snip_global --pruning_fraction=0.5
