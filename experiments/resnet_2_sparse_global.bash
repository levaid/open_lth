#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
for i in {1..10};
do
	python open_lth.py lottery --model_name=cifar_resnet_38_64 --levels=8 --dataset_name=cifar10 --batch_size=128  --model_init=kaiming_normal --batchnorm_init=uniform --optimizer_name=sgd --lr=0.1 --momentum=0.1 --training_steps=80ep --pruning_strategy=sparse_global --pruning_fraction=0.50 --replicate=$i --milestone_steps=60ep --gamma=0.1 --weight_decay=0.0001;
done
