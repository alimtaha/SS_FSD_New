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
export mode="Thesis_Pics"
export debug="False"
export full_depth_resnet="True"
export load_checkpoint="False"
export load_depth_checkpoint="True"
export depth_only="False"
export no_classes=19

export semi_checkpoint_path="/mnt/Dataset/Logs/SSL/CPS/Semi/All_Semi-Supervision_Ratio16_22-May_16-39-nodebs2-tep100-lr0.02-maxdepth80_newcrfs/epoch-best_loss.pth"
export depth_checkpoint_path="/mnt/Dataset/Logs/SSL/CPS/Semi/All-Labels-Concat-Semi-Supervision-FullResnet_Ratio16_14-May_17-39-nodebs2-tep100-lr0.02-maxdepth80_newcrfs/epoch-best_loss.pth"

export volna="/mnt/Dataset/city/"
export OUTPUT_PATH='/mnt/Dataset/Logs/SSL/CPS/Thesis_Pics/'
export snapshot_dir='/mnt/Dataset/Logs/SSL/CPS/Thesis_Pics/'

python thesis_pics.py  #-m torch.distributed.launch --nproc_per_node=$NGPUS  #https://pytorch.org/docs/stable/elastic/run.html
export TARGET_DEVICE=$[$NGPUS-1]                                    #https://pytorch.org/docs/stable/distributed.html - scroll down for explanation of launch