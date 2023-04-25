#!/usr/bin/env bash
nvidia-smi

export NGPUS=1
export learning_rate=0.002
export batch_size=2
export snapshot_iter=2
export epochs=35
export ratio=16
export WORLD_SIZE=$batch_size
export CPU_DIST_ONLY='True'
export debug="False"

export volna="/home/extraspace/Datasets/Datasets/cityscapes/city/"
export OUTPUT_PATH="/home/extraspace/Runs/CPS/Semi/1-$ratio/"
export snapshot_dir="/home/extraspace/Runs/CPS/Semi/1-$ratio/depth_concat/$learning_rate_$epochs"


python -m torch.distributed.launch --nnodes=1 --nproc_per_node=$WORLD_SIZE train_depth_concat_cpu.py  #-m torch.distributed.launch --nproc_per_node=$NGPUS  #https://pytorch.org/docs/stable/elastic/run.html
#export TARGET_DEVICE=$[$NGPUS-1]                                    #https://pytorch.org/docs/stable/distributed.html - scroll down for explanation of launch
#python eval.py -e 110-137 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results

# following is the command for debug
# export NGPUS=1
# export learning_rate=0.0025
# export batch_size=1
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --debug 1