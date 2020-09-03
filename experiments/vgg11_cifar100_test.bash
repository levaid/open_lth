#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python ../open_lth.py lottery --model_name=cifar_vgg_11 --levels=9 \
    --do_not_augment --dataset_name=cifar100 --batch_size=128  --model_init=kaiming_normal \
    --batchnorm_init=uniform --optimizer_name=sgd --lr=0.1 --momentum=0.1 --training_steps=10ep \
    --pruning_strategy=snip_global --pruning_fraction=0.5 --replicate=1 --milestone_steps=10ep --gamma=0.1 --weight_decay=0.0001 --num_workers=1;

