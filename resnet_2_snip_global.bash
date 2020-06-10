#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
python open_lth.py lottery --model_name=cifar_resnet_38_64 --levels=6 --dataset_name=cifar10 --batch_size=128  --model_init=kaiming_normal --batchnorm_init=uniform --optimizer_name=sgd --lr=0.1 --momentum=0.1 --training_steps=1ep --pruning_strategy=snip_global --pruning_fraction=0.49 --replicate=2 --milestone_steps=60ep --gamma=0.1 --weight_decay=0.0001
