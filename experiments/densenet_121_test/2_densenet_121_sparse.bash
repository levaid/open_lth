#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
pruning_strategy="sparse_global"

for r in 1 2 3 4; do
    python ../../open_lth.py lottery --model_name=densenet_121_tinyimagenet --levels=10 --dataset_name=tinyimagenet --batch_size=128  --model_init=kaiming_normal --batchnorm_init=uniform --optimizer_name=sgd --lr=0.1 --momentum=0.9 --training_steps=120ep --pruning_strategy=$pruning_strategy --pruning_fraction=0.4 --replicate=$r --milestone_steps=80ep,100ep,115ep --gamma=0.1 --weight_decay=0.0001 --num_workers=1;
done

#1e = 391it
