# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import time
from datetime import datetime as dt
import numpy as np
from easydict import EasyDict as edict
import argparse
import socket

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


C = edict()
config = C
cfg = C

C.aws = False if socket.gethostname() in ['wmgubws06.wmgds.wmg.warwick.ac.uk', 'wmgubws05.wmgds.wmg.warwick.ac.uk'] else True

C.seed = 12345

remoteip = os.popen('pwd').read()
if os.getenv('volna') is not None:
    C.volna = os.environ['volna']
else:
    if C.aws:
    # the path to the data dir.
        C.volna = '/mnt/Dataset/city'
    else:
        C.volna = '/home/extraspace/Datasets/Datasets/cityscapes/city'


"""please config ROOT_dir and user when u first using"""
C.repo_name = 'TorchSemiSeg'
C.abs_dir = osp.realpath(".")
C.this_dir = C.abs_dir.split(osp.sep)[-1]
C.weak_labels = False
C.labeled_ratio = int(os.environ['ratio'])

C.is_test = False
C.fix_bias = True
C.bn_eps = 1e-5
C.bn_momentum = 0.1

C.cps_weight = 5
C.contrast_weight = 5
C.sup_contrast_weight = 1

"""Cross Attention Configuration"""
C.cross_att_mode = 'dual_patch_patch'
C.cross_attention = True if C.cross_att_mode in [ 
    'image_patch_patch',
    'dual_patch_patch',
    'image_token_patch',
    'dual_token_patch',
    'token_token'] else False

"""Cutmix Config"""
C.cutmix_mask_prop_range = (0.25, 0.5)
C.cutmix_boxmask_n_boxes = 1
C.cutmix_boxmask_fixed_aspect_ratio = False
C.cutmix_boxmask_by_size = False
C.cutmix_boxmask_outside_bounds = False
C.cutmix_boxmask_no_invert = False

"""DepthMix Config"""
C.depthmix = False
C.depthmix_mask_prop_range = (0.99, 1.0)
C.depthmix_boxmask_n_boxes = 1
C.depthmix_boxmask_fixed_aspect_ratio = False
C.depthmix_boxmask_by_size = False
C.depthmix_boxmask_outside_bounds = False
C.depthmix_boxmask_no_invert = False

"""Image Config"""
C.num_classes = 2                              # need to change for training free space detection
C.background = 100  # background changed to the class for the padding when cropping
C.image_mean = np.array([0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
C.image_std = np.array([0.229, 0.224, 0.225])
C.dimage_mean = 24.188  #Generated using full train & val & test sets on NewCRFs 80 Depth Maps
C.dimage_std = 19.530
# C.dimage_mean = 29.091  #Generated using full train & val & test sets on NewCRFs 256 Depth Maps
# C.dimage_std = 31.799
C.image_height = 620
C.image_width = 620
# if ratio is 8, becomes 371 (// returns the int of the division)
C.num_train_imgs = 2975 // C.labeled_ratio
C.num_eval_imgs = 500
C.num_unsup_imgs = 2975 - C.num_train_imgs  # if ratio is 8, becomes 2604
C.crop_pos = None
C.img_shape_h = 1024
C.img_shape_w = 2048
C.pixelation_factor = 8     #trialled 4 - implement randomised choice..?
C.lowres_factor = 8         #implement randomised choice..?
C.gaussian_kernel = [9,9]
C.sup_contrast = False
C.ignore_label = 100

"""Train Config"""
if os.getenv('learning_rate'):
    C.lr = float(os.environ['learning_rate'])
else:
    C.lr = 0.002

if os.getenv('batch_size'):
    C.batch_size = int(os.environ['batch_size'])
else:
    C.batch_size = 4

C.optimiser = str(os.environ['optim']) if os.getenv('optim') else 'SGD'
C.lr_power = 0.9
C.adam_betas = (0.9, 0.98)
C.momentum = 0.8
C.optim_params = C.momentum if C.optimiser == 'SGD' else C.adam_betas
C.weight_decay = 0.0001 #1e-4
C.attn_lr_factor = 8
C.head_lr_factor = 2
C.attn_heads = 4
C.checkpoint_path = '/mnt/Dataset/Logs/SSL/CPS/Semi/Semi-Supervision_Ratio16_22-Apr_21-23-nodebs2-tep35-lr0.002-maxdepth80_newcrfs/epoch-best_loss.pth'

# 35 #122.8 epochs to equal number of iterations for supervised baseline.
# original - 137
C.nepochs = int(os.environ['epochs'])
C.max_samples = max(C.num_train_imgs, C.num_unsup_imgs)
C.cold_start = 0
C.niters_per_epoch = C.max_samples // C.batch_size  # 2604 / 2 for me
C.fully_sup_iters = C.num_train_imgs // C.batch_size

C.num_workers = 8

print(
    bcolors.WARNING +
    f'\n\n\n-------NUMBER OF WORKERS SET TO {C.num_workers}!!!!! CHANGE BACK FOR GPU TRAINING IF 0-------\n\n\n' +
    bcolors.WARNING)

if C.weak_labels:
    print(
    bcolors.FAIL + bcolors.BOLD + 
    f'\n\n\n-------USING WEAK LABELS!!!!!-------\n\n\n' +
    bcolors.BOLD + bcolors.FAIL)

# [1, 1.5, 1.75, 2.0]#[0.5, 0.75, 1, 1.5, 1.75, 2.0]
C.train_scale_array = None

"""Eval Config"""
C.eval_iter = 30
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1, ]  # 0.5, 0.75, 1, 1.5, 1.75
C.eval_flip = False
C.eval_base_size = 800
C.eval_crop_size = [1024, 2048]

"""Display Config"""
if os.getenv('snapshot_iter'):
    C.snapshot_iter = int(os.environ['snapshot_iter'])
else:
    C.snapshot_iter = 2
C.record_info_iter = 20
C.display_iter = 50
C.warm_up_epoch = 0  # experiment with warm up epoch
C.validate_every = 695  # C.max_samples
C.embed_every = C.validate_every*4

#to unify all scripts into one master script, modes are used
#modes are depth_concat, contrastive_depth_concat, crossattention_depth_concat - more to be added (semi-super, fully-super, depth-append)
C.mode = os.environ['mode'] + '_Ratio' + os.environ['ratio']         

C.depth_ckpt = 'newcrfs_80.0_model-44982-best_d1_0.95066/'
C.max_d = 80 if C.depth_ckpt == 'newcrfs_80.0_model-44982-best_d1_0.95066/' else 256


run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{C.batch_size}-tep{C.nepochs}-lr{C.lr}-maxdepth{C.max_d}_{C.depth_ckpt.split('_')[0]}"
name = f"{C.mode}_'Pretrained-{os.environ['load_checkpoint']}_{run_id}"

C.log_dir = os.path.join(os.environ['snapshot_dir'], name)
C.tb_dir = C.log_dir

#future config to be added to allow all labels

#C.root_dir = C.abs_dir[:C.abs_dir.index(C.repo_name) + len(C.repo_name)]

if C.aws:
    C.root_dir = '/home/ubuntu/SS_FSD_New/SS_FSD_New/CPS/TorchSemiSeg'
    
    """Data Dir and Weight Dir"""
    C.dataset_path = C.volna
    C.img_root_folder = '/mnt/Dataset/city'
    C.gt_root_folder = '/mnt/Dataset/city'
    C.pretrained_model = '/mnt/Dataset/models/backbones/resnet50_v1c.pth'
    #C.log_dir_link = osp.join(C.abs_dir, 'log')
else:
    C.root_dir = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/SS_FSD_New/CPS/TorchSemiSeg'
    """Data Dir and Weight Dir"""
    C.dataset_path = C.volna
    C.img_root_folder = C.dataset_path
    C.gt_root_folder = C.dataset_path
    C.pretrained_model = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/CPS/TorchSemiSeg/DATA/pytorch-weight/resnet50_v1c.pth'
    #C.log_dir_link = osp.join(C.abs_dir, 'log')

# snapshot dir that stores checkpoints
# if os.getenv('snapshot_dir'):
#     C.snapshot_dir = osp.join(os.environ['snapshot_dir'], "snapshot")
# else:

C.snapshot_dir = osp.abspath(osp.join(C.log_dir, "snapshot"))

exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
#C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
#C.link_val_log_file = C.log_dir + '/val_last.log'

"""Path Config"""
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(osp.join(C.root_dir, 'furnace'))


''' Experiments Setting '''
C.train_source = osp.join(
    C.dataset_path, "config_new/subset_train/train_aug_labeled_1-{}.txt".format(C.labeled_ratio))
C.train_source = osp.join(
    C.dataset_path, "config_new/subset_train/train_aug_labeled_1-{}.txt".format(C.labeled_ratio))
C.unsup_source = osp.join(
    C.dataset_path, "config_new/subset_train/train_aug_unlabeled_1-{}.txt".format(C.labeled_ratio))
C.unsup_source_1 = osp.join(
    C.dataset_path,
    "config_new/subset_train/train_aug_unlabeled_1-{}_shuffle.txt".format(
        C.labeled_ratio))
C.eval_source = osp.join(C.dataset_path, "config_new/val.txt")
C.test_source = osp.join(C.dataset_path, "config_new/test.txt")
C.demo_source = osp.join(C.dataset_path, "config_new/demo.txt")