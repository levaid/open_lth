#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
pruning_strategy="sparse_global"

for r in 3 4; do
    for l in 1 2 3 4 5 6 7 8 9 10 11; do
        for e in 60 30 10 5 2 1; do
            python ../../open_lth.py lottery --model_name=cifar_resnet_20 --levels=$l --do_not_augment --dataset_name=cifar10 --batch_size=128  --model_init=kaiming_normal --batchnorm_init=uniform --optimizer_name=sgd --lr=0.1 --momentum=0.9 --training_steps="${e}ep" --pruning_strategy=$pruning_strategy --pruning_fraction=0.5 --replicate=$r --milestone_steps=40ep --gamma=0.1 --weight_decay=0.0001 --posttrain_steps=60ep --num_workers=1;
        done
    done
done

#1e = 391it
