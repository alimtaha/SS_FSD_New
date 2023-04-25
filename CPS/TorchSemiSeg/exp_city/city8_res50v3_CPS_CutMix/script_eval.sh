#!/usr/bin/env bash
nvidia-smi

export volna="/media/taha_a/T7/Datasets/cityscapes/city"
export OUTPUT_PATH="/media/taha_a/T7/Datasets/cityscapes/outputs/city"
export snapshot_dir=$OUTPUT_PATH/snapshot

export NGPUS=1
export learning_rate=0.002
export batch_size=2
export snapshot_iter=2


#python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py #https://pytorch.org/docs/stable/elastic/run.html
export TARGET_DEVICE=$[$NGPUS-1]                                    #https://pytorch.org/docs/stable/distributed.html - scroll down for explanation of launch
python eval.py -e '/media/taha_a/T7/Datasets/cityscapes/outputs/city/snapshot/snapshot/2022-06-22-12-02-55-epoch-16.pth' -d 0 --save_path $OUTPUT_PATH/results

# following is the command for debug
# export NGPUS=1
# export learning_rate=0.0025
# export batch_size=1
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --debug 1