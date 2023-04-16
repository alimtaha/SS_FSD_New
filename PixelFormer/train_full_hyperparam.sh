#!/bin/bash

for depth in 80 256
do
    export H_DATASET='multi_new_depth_inf'
    export H_WIDTH=352
    export H_LR=0.00001
    export H_MAXDEPTH=$depth
    export H_HEIGHT=1024
    python hyperparam/train_hyperparam_full.py hyperparam/hyperparams_full_arguments_train_cityscapes.txt
done

