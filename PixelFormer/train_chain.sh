#!/bin/bash

for lr in 0.00001 0.00002 0.00004
do    
    for width in 1120 352 704
    do
        for depth in 80 256
        do
            export H_DATASET='multi_new_depth_inf'
            export H_WIDTH=$width
            export H_LR=$lr
            export H_MAXDEPTH=$depth
            python hyperparam/train_hyperparam.py hyperparam/hyperparams_arguments_train_cityscapes.txt
        done
    done
done

for lr in 0.00001 0.00002 0.00004
do    
    for width in 352 704 1120
    do
        export H_DATASET='depth_80'
        export H_WIDTH=$width
        export H_LR=$lr
        export H_MAXDEPTH=80
        python hyperparam/train_hyperparam.py hyperparam/hyperparams_arguments_train_cityscapes.txt
    done
done