#!/usr/bin/env bash
nvidia-smi

export NGPUS=1
export learning_rate=0.02
export batch_size=2
export snapshot_iter=2
export epochs=107
export ratio=8
export CPU_DIST_ONLY='False'
export WORLD_SIZE=1
export mode="All-Labels-Concat-Semi-Supervision-FullResnet"
export debug="False"
export full_depth_resnet="True"
export load_checkpoint="True"
export load_depth_checkpoint="True"
export depth_only="False"
export no_classes=19

export checkpoint_path="/mnt/Dataset/Logs/SSL/CPS/Semi/All_Semi-Supervision_Ratio8_25-May_15-20-nodebs2-tep107-lr0.02-maxdepth80_newcrfs/epoch-best_loss.pth"
export depth_checkpoint_path="/mnt/Dataset/Logs/SSL/CPS/Semi/All-Labels-Concat-Semi-Supervision-FullResnet_Ratio8__DepthOnly_25-May_15-26-nodebs2-tep107-lr0.02-maxdepth80_newcrfs/epoch-best_loss.pth"

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