export CUDA_VISIBLE_DEVICES=2
for i in {1..5};
do
    python initpruning_vgg.py 'snip' '11';
done