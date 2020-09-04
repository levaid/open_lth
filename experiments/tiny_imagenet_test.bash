python ../open_lth.py lottery --model_name=imagenet_resnet_18 --levels=9 \
    --do_not_augment --dataset_name=tinyimagenet --batch_size=64  --model_init=kaiming_normal \
    --batchnorm_init=uniform --optimizer_name=sgd --lr=0.1 --momentum=0.9 --training_steps=5ep \
    --pruning_strategy=snip_global --pruning_fraction=0.5 --replicate=1 --milestone_steps=5ep --gamma=0.1 --weight_decay=0.0001 --num_workers=2; 
