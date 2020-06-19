export CUDA_VISIBLE_DEVICES=4
for i in {1..5};
do
    python initpruning_vgg.py 'snip' '16';
done