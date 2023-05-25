#!/usr/bin/env bash
nvidia-smi

export NGPUS=1
export learning_rate=0.002
export batch_size=2
export snapshot_iter=2
export epochs=107
export ratio=8
export CPU_DIST_ONLY='False'
export WORLD_SIZE=1
export mode="Concat-Semi-Supervision-FullResnet"
export debug="False"
export full_depth_resnet="True"
export load_checkpoint="False"
export load_depth_checkpoint="False"
export depth_only="False"
export no_classes=2

export checkpoint_path="/mnt/Dataset/Logs/SSL/CPS/Semi/Semi-Supervision_Ratio8_13-May_13-25-nodebs2-tep40-lr0.002-maxdepth80_newcrfs/epoch-best_loss.pth"
export depth_checkpoint_path="/mnt/Dataset/Logs/SSL/CPS/Semi/Concat-Semi-Supervision-FullResnet_Ratio8__DepthOnly_22-May_03-04-nodebs2-tep35-lr0.002-maxdepth80_newcrfs/epoch-best_loss.pth"

export volna="/mnt/Dataset/city/"
export OUTPUT_PATH='/mnt/Dataset/Logs/SSL/CPS/Semi/'
export snapshot_dir='/mnt/Dataset/Logs/SSL/CPS/Semi/'

python train_depth_concat_fullresnet.py  #-m torch.distributed.launch --nproc_per_node=$NGPUS  #https://pytorch.org/docs/stable/elastic/run.html
export TARGET_DEVICE=$[$NGPUS-1]                                    #https://pytorch.org/docs/stable/distributed.html - scroll down for explanation of launch
#python eval.py -e 110-137 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results

# following is the command for debug
# export NGPUS=1
# export learning_rate=0.0025
# export batch_size=1
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --debug 1