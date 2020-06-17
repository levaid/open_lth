#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
for i in {1..10}; 
do
    python open_lth.py lottery --model_name=cifar_vgg_11 --levels=9 --do_not_augment --dataset_name=cifar10 --batch_size=128  --model_init=kaiming_normal --batchnorm_init=uniform --optimizer_name=sgd --lr=0.1 --momentum=0.1 --training_steps=60ep --pruning_strategy=sparse_global --pruning_fraction=0.5 --replicate=$i --milestone_steps=40ep --gamma=0.1 --weight_decay=0.0001;
done
