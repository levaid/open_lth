#!/bin/bash

python open_lth.py lottery --model_name=mnist_lenet_300_100 --levels=4 --rewinding_steps=3ep --transformation_seed=49 --dataset_name=mnist --batch_size=128  --model_init=kaiming_normal --batchnorm_init=uniform --optimizer_name=sgd --lr=0.01 --training_steps=10ep --pruning_strategy=snip_global --pruning_fraction=0.5
