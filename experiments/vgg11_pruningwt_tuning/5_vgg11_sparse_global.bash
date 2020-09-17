#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
strategy='sparse_global'

for r in 1 2 3 4 5; do
        python ../../open_lth.py lottery --model_name=cifar_vgg_11 --levels=8 --dataset_name=cifar10 --batch_size=128  --model_init=kaiming_normal --batchnorm_init=uniform --optimizer_name=sgd --lr=0.1 --momentum=0.9 --training_steps=100ep --pruning_strategy=$strategy --pruning_fraction=0.5 --replicate=$r --milestone_steps=60ep,85ep --gamma=0.1 --weight_decay=0.0001 --num_workers=1;
done
