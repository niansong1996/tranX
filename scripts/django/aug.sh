#!/bin/bash

source activate py2torch3cuda9

seed=${1:-0}
train_file="train.bin"

python exp.py \
    --seed ${seed} \
    --mode augmentation \
    --batch_size 10 \
    --asdl_file asdl/lang/py/py_asdl.txt \
    --train_file data/django/${train_file} \
    --augmentation reconstructor \
    --label_sample_ratio 0.5 \
    --load_augmentation_model model.bin 
