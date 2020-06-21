export CUDA_VISIBLE_DEVICES=7
for i in {1..5};
do
    python initpruning_vgg.py 'sparse' '11';
done
