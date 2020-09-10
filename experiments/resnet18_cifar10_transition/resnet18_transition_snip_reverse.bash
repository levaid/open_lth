#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

for r in 3 4;
do
    for e in 60 50 40 30 20 15 10 8 6 5 4 3 2 1; 
    do
        python ../../open_lth.py lottery --model_name=cifar_resnet_20 --levels=11 --do_not_augment --dataset_name=cifar10 --batch_size=128  --model_init=kaiming_normal --batchnorm_init=uniform --optimizer_name=sgd --lr=0.1 --momentum=0.9 --training_steps="${e}ep" --pruning_strategy=snip_global --pruning_fraction=0.4 --replicate=$r --milestone_steps=40ep --gamma=0.1 --weight_decay=0.0001 --posttrain_steps=60ep --num_workers=1;
    done

done
