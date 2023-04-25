#!/usr/bin/env bash
nvidia-smi

export NGPUS=1
export learning_rate=0.002
export batch_size=2
export snapshot_iter=2
export epochs=35
export ratio=16
export CPU_DIST_ONLY='False'
export WORLD_SIZE=1
export debug="False"
export optim='SGD'

export volna="/mnt/Dataset/city/"
export OUTPUT_PATH='/mnt/Dataset/Logs/SSL/CPS/Semi/'
export snapshot_dir='/mnt/Dataset/Logs/SSL/CPS/Semi/'
export RAY_PICKLE_VERBOSE_DEBUG=1

python train_crossattention_depth_concat.py  #-m torch.distributed.launch --nproc_per_node=$NGPUS  #https://pytorch.org/docs/stable/elastic/run.html
export TARGET_DEVICE=$[$NGPUS-1]                                    #https://pytorch.org/docs/stable/distributed.html - scroll down for explanation of launch
#python eval.py -e 110-137 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results

# following is the command for debug
# export NGPUS=1
# export learning_rate=0.0025
# export batch_size=1
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --debug 1