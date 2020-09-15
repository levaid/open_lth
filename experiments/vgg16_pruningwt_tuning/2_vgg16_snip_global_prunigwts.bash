#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
strategy='snip_global'

for r in 3 4; do
    for weight in 3 2 1.5 1.25 1 0.75 0.5; do 
        python ../../open_lth.py lottery --model_name=cifar_vgg_16 --levels=13 --dataset_name=cifar10 --batch_size=128  --model_init=kaiming_normal --batchnorm_init=uniform --optimizer_name=sgd --lr=0.1 --momentum=0.9 --training_steps=100ep --pruning_strategy=$strategy --pruning_fraction=0.5 --replicate=$r --milestone_steps=60ep --gamma=0.1 --weight_decay=0.0001 --pruning_gradient_weight=$weight --num_workers=1;
    done
done
