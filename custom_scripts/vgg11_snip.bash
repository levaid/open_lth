export CUDA_VISIBLE_DEVICES=6
for i in {1..5};
do
    python initpruning_vgg.py 'snip' '11';
done
