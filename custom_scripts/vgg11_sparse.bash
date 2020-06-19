export CUDA_VISIBLE_DEVICES=3
for i in {1..5};
do
    python initpruning_vgg.py 'sparse' '11';
done