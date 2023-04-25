#!/bin/bash

for lr in 0.00001 0.00002 0.00004
do
    for depth in 80 256
    do
        export H_DATASET='multi_new_depth_inf'
        export H_WIDTH=352
        export H_LR=$lr
        export H_MAXDEPTH=$depth
        export H_HEIGHT=1024
        python hyperparam/train_hyperparam.py hyperparam/hyperparams_arguments_train_cityscapes.txt
    done
done

