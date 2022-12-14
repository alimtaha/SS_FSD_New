#!/usr/bin/env bash
nvidia-smi

export volna="TorchSemiSeg/DATA"
export OUTPUT_PATH="path to your output dir"
export snapshot_dir=$OUTPUT_PATH/snapshot

export NGPUS=8
export learning_rate=0.02
export batch_size=8
export snapshot_iter=5

python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py #https://pytorch.org/docs/stable/elastic/run.html
export TARGET_DEVICE=$[$NGPUS-1]                                    #https://pytorch.org/docs/stable/distributed.html - scroll down for explanation of launch
python eval.py -e 110-137 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results

# following is the command for debug
# export NGPUS=1
# export learning_rate=0.0025
# export batch_size=1
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --debug 1