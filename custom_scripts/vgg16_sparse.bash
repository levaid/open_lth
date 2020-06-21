export CUDA_VISIBLE_DEVICES=1
for i in {1..5};
do
    python initpruning_vgg.py 'sparse' '16';
done
