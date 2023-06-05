from __future__ import division
import os.path as osp
import sys
import os
sys.path.append(os.getcwd() + '/../../..')
sys.path.append(os.getcwd() + '/../..')
sys.path.append(os.getcwd() + '/..')
from custom_collate import SegCollate
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from furnace.seg_opr.metric import hist_info, compute_score, compute_score_recall_precision
from furnace.engine.evaluator import Evaluator
from furnace.utils.pyt_utils import load_model
from furnace.seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d, bce2d
from furnace.engine.engine import Engine
from furnace.engine.lr_policy import WarmUpPolyLR
from furnace.utils.visualize import print_iou, show_img, print_pr
from furnace.utils.init_func import init_weight, group_weight
from furnace.utils.img_utils import generate_random_uns_crop_pos
import random
import cv2
import pandas as pd
from network import Network
from dataloader_depth_concat import CityScape as CityScape_All
from dataloader_depth_concat import TrainValPre as TrainValPre_All
from dataloader_all import CityScape, TrainValPre
from network_depth_concat import NetworkFullResnet
from config import config
from matplotlib import colors
from PIL import Image
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Parameter
import torchvision.utils
import torch
import numpy as np
from datetime import datetime as dt
from tqdm import tqdm
import argparse
import math
import time
import uuid
import os

image_layer_tuple = (
    "branch1.backbone",
    "branch1.head.reduce",
    "branch1.head.aspp.map_convs",
    "branch1.head.aspp.pool_u2pl",
    "branch1.head.aspp.red_conv",
    "branch2.backbone",
    "branch2.head.reduce",
    "branch2.head.aspp.map_convs",
    "branch2.head.aspp.pool_u2pl",
    "branch2.head.aspp.red_conv",
)

depth_layer_tuple = (
    "branch1.depth_backbone",
    "branch1.head.depth_reduce",
    # "branch1.head.last_conv",
    "branch1.head.aspp.depth_map_convs",
    "branch1.head.aspp.pool_depth",
    "branch1.head.aspp.depth_red_conv",
    "branch1.classifier",       #may need removal
    "branch2.depth_backbone",
    "branch2.head.depth_reduce",
    # "branch2.head.last_conv",
    "branch2.head.aspp.depth_map_convs",
    "branch2.head.aspp.pool_depth",
    "branch2.head.aspp.depth_red_conv",
    "branch2.classifier",       #may need removal
)

def compute_metric(results):
    hist = np.zeros((config.num_classes, config.num_classes))
    correct = 0
    labeled = 0
    count = 0
    for d in results:
        hist += d['hist']
        correct += d['correct']
        labeled += d['labeled']
        count += 1

    p, mean_p, r, mean_r, mean_p_no_back, mean_r_no_back = compute_score_recall_precision(hist, correct, labeled)
    iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct,
                                                   labeled)
    # changed from the variable dataset to the class directly so this function
    # can now be called without first initialising the eval file

    return iu, mean_IU, _, mean_pixel_acc, p, mean_p, r, mean_r, mean_p_no_back, mean_r_no_back

def viz_image(imgs, gts, pred, step, epoch, name, logger, step_test=None):
    image_viz = (imgs[0,
                      :3,
                      :,
                      :].squeeze().cpu().numpy() * np.expand_dims(np.expand_dims(config.image_std,
                                                                                 axis=1),
                                                                  axis=2) + np.expand_dims(np.expand_dims(config.image_mean,
                                                                                                          axis=1),
                                                                                           axis=2)) * 255.0
    image_viz = image_viz.transpose(1, 2, 0)
    depth_image = imgs[0, 3:, :, :].squeeze().cpu(
    ).numpy() * config.dimage_std + config.dimage_mean
    label = np.asarray(gts[0, :, :].squeeze().cpu(), dtype=np.uint8)
    clean = np.zeros(label.shape)
    pred_viz = torch.argmax(pred[0, :, :, :], dim=0).cpu()
    pred_viz = np.array(pred_viz, np.uint8)
    comp_img = show_img(CityScape.get_class_colors(), config.background, image_viz, clean,  # image size is 720 x 2190 x 3
                        label, pred_viz)
    # logger needs the RGB dimension as the first one
    comp_img = comp_img.transpose(2, 0, 1)
    if step_test is None:
        logger.add_image(
            f'Training/Epoch_{epoch}/Image_Pred_GT_{step}_{name}',
            comp_img,
            step)
    else:
        logger.add_image(
            f'Val/Epoch_{epoch}/Val_Step_{step/config.validate_every}/Image_Pred_GT_{step_test}_{name}',
            comp_img,
            step)

depth_path = os.path.join(config.tb_dir, 'depth' + str(config.depth_checkpoint_path.split('/')[-2]))
semi_path = os.path.join(config.tb_dir, 'images' + str(config.semi_checkpoint_path.split('/')[-2]))

tb_dir = config.tb_dir
depth_logger = SummaryWriter(
    log_dir= depth_path
    )
image_logger = SummaryWriter(
    log_dir= semi_path
    )

parser = argparse.ArgumentParser()
os.environ['MASTER_PORT'] = '169711'

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True  # Changed to False due to error

    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}

    pixel_num = 500 * config.batch_size // engine.world_size
    criterion = ProbOhemCrossEntropy2d(ignore_label=config.ignore_label, thresh=0.7, min_kept=pixel_num, use_weight=False) # NUMBER CHANGED TO 5000 from 50000 due to reduction in number of labels since only road labels valid
     

    model_semi = Network(config.num_classes, criterion=criterion,  # change number of classes to free space only
                    pretrained_model=config.pretrained_model,
                    norm_layer=nn.BatchNorm2d)  # need to change norm_layer to nn.BatchNorm2d since BatchNorm2d is derived from the furnace package and doesn't seem to work, it's only needed for syncing batches across multiple GPU, may be needed later


    trainval_pre = TrainValPre(config.image_mean, config.image_std)
    test_dataset = CityScape(data_setting, 'trainval', trainval_pre)

    test_loader = data.DataLoader(test_dataset,
                                  batch_size=1,
                                  num_workers=config.num_workers,
                                  drop_last=True,
                                  shuffle=True,
                                  pin_memory=True,
                                  sampler=None)

    
    model_depth = NetworkFullResnet(config.num_classes, criterion=criterion,  # change number of classes to free space only
                    pretrained_model=config.pretrained_model,
                    norm_layer=nn.BatchNorm2d,
                    full_depth_resnet=True,
                    depth_only=config.depth_only) 

    trainval_pre_all = TrainValPre_All(config.image_mean, config.image_std, config.dimage_mean, config.dimage_std)
    test_dataset_all = CityScape_All(data_setting, 'trainval', trainval_pre_all)

    test_loader_all = data.DataLoader(test_dataset_all,
                                  batch_size=1,
                                  num_workers=config.num_workers,
                                  drop_last=True,
                                  shuffle=True,
                                  pin_memory=True,
                                  sampler=None)
    
    #Loading depth model
    depth_state_dict = torch.load(config.depth_checkpoint_path) #Change
    own_state = model_depth.state_dict()
    print(own_state['branch1.classifier.bias'].data, depth_state_dict['model']['branch1.classifier.bias'].data)
    for name, param in depth_state_dict['model'].items():
        #if (name in own_state):#('branch1.head.depth') or name.startswith('branch2.head.last_conv'):
        param = param.data
        own_state[name].copy_(param)
        #    image_layers_loaded.append(name)
        #else:
        #    continue

    print('Checkpoint loaded for Depth Model: ', config.depth_checkpoint_path)

    #Loading image model

    print('Loading: ', config.semi_checkpoint_path)
    semi_state_dict = torch.load(config.semi_checkpoint_path) #Change
    own_state = model_semi.state_dict()
    print(own_state['branch1.classifier.bias'].data, semi_state_dict['model']['branch1.classifier.bias'].data)
    for name, param in semi_state_dict['model'].items():
        #if (name in own_state):#('branch1.head.depth') or name.startswith('branch2.head.last_conv'):
        param = param.data
        own_state[name].copy_(param)
        #else:
        #    continue

    print('Checkpoint loaded for Image Model: ', config.semi_checkpoint_path)

    device = torch.device("cuda") #Change

    model_depth.to(device)

    model_depth.eval()

    all_results_depth = []

    results_dict_export_depth = {'frame': ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign',
                'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'mIoU', 'mean_accuracy', 'mean_p', 'mean_r', 'loss']}

    #depth evaluation
    with torch.no_grad():

        for batch_test in tqdm(
                test_loader_all,
                desc=f"Loop: Validation",
                total=len(test_loader_all)):

            imgs_test = batch_test['data'].to(device)
            gts_test = batch_test['label'].to(device)

            pred_test, _, v3_feats = model_depth.branch1(imgs_test)

            loss_sup_test = criterion(pred_test, gts_test)
            pred_test_max = torch.argmax(pred_test[0, :, :, :], dim=0).long().cpu().numpy() 

            hist_tmp, labeled_tmp, correct_tmp = hist_info(
                config.num_classes, pred_test_max, gts_test[0, :, :].cpu().numpy())

            results_dict = {
                'hist': hist_tmp,
                'labeled': labeled_tmp,
                'correct': correct_tmp}
            all_results_depth.append(results_dict)

            viz_image(
                imgs_test,
                gts_test,
                pred_test,
                0,
                0,
                batch_test['fn'][0],
                depth_logger,
                0)

            iu, mean_IU, _, mean_pixel_acc, p, mean_p, r, mean_r, mean_p_no_back, mean_r_no_back = compute_metric(all_results_depth)

            all_results_depth = []

            frame_name = batch_test['fn'][0]
            results_dict_export_depth[frame_name] = [iu[0], iu[1], iu[2], iu[3], iu[4], iu[5], iu[6], iu[7], iu[8], iu[9], iu[10], iu[11], iu[12], iu[13], iu[14], iu[15], iu[16], iu[17], iu[18], mean_IU, mean_pixel_acc, mean_p, mean_r, loss_sup_test]

    model_depth.cpu()

    depth = pd.DataFrame.from_dict(results_dict_export_depth, orient='index')

    depth.to_csv(depth_path + 'results.csv')

    del model_depth

    model_semi.to(device)

    model_semi.eval()

    all_results_semi = []

    results_dict_export_semi = {'frame': 
            ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign',
            'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'mIoU', 
            'mean_accuracy', 'mean_p', 'mean_r', 'loss']}

    #semi evaluation
    with torch.no_grad():

        for batch_test in tqdm(
                test_loader,
                desc=f"Loop: Validation",
                total=len(test_loader)):

            imgs_test = batch_test['data'].to(device)
            gts_test = batch_test['label'].to(device)

            pred_test, _, v3_feats = model_semi.branch1(imgs_test)

            loss_sup_test = criterion(pred_test, gts_test)
            pred_test_max = torch.argmax(pred_test[0, :, :, :], dim=0).long().cpu().numpy() 

            hist_tmp, labeled_tmp, correct_tmp = hist_info(
                config.num_classes, pred_test_max, gts_test[0, :, :].cpu().numpy())

            results_dict = {
                'hist': hist_tmp,
                'labeled': labeled_tmp,
                'correct': correct_tmp}
            all_results_semi.append(results_dict)

            viz_image(
                imgs_test,
                gts_test,
                pred_test,
                0,
                0,
                batch_test['fn'][0],
                image_logger,
                0)

            iu, mean_IU, _, mean_pixel_acc, p, mean_p, r, mean_r, mean_p_no_back, mean_r_no_back = compute_metric(all_results_semi)

            all_results_semi = []

            frame_name = batch_test['fn'][0]
            results_dict_export_semi[frame_name] = [iu[0], iu[1], iu[2], iu[3], iu[4], iu[5], iu[6], iu[7], iu[8], iu[9], iu[10], iu[11], iu[12], iu[13], iu[14], iu[15], iu[16], iu[17], iu[18], mean_IU, mean_pixel_acc, mean_p, mean_r, loss_sup_test]

    semi = pd.DataFrame.from_dict(results_dict_export_semi, orient='index')

    semi.to_csv(semi_path + 'results.csv')

