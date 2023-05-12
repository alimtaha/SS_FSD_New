from __future__ import division
import os.path as osp
import sys
import os
sys.path.append(os.getcwd() + '/../../..')
sys.path.append(os.getcwd() + '/../..')
sys.path.append(os.getcwd() + '/..')
from custom_collate import SegCollate
import mask_gen_depth
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
from eval_depth_concat import SegEvaluator
from dataloader_depth_concat import CityScape, get_train_loader
from network_depth_concat import Network, NetworkFullResnet, count_params
from config import config
from matplotlib import colors
from PIL import Image
from dataloader_depth_concat import TrainValPre
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

'''NEEED TO UPDATE VALIDATION AND EVAL FILE AND VAL PRE ETC TO INCLIDE DEPTH VALUES AND NEW FUNCTIONS'''

#from furnace.seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d

PROJECT = 'CPS'
if config.weak_labels:
    experiment_name = 'weak_labels' + '_ConcatD_' + str(config.nepochs) + 'E_SS' + str(config.labeled_ratio) + \
    '_L' + str(config.lr) + str(config.image_height) + 'size'
else:
    experiment_name = '_ConcatD_' + str(config.nepochs) + 'E_SS' + str(config.labeled_ratio) + \
    '_L' + str(config.lr) + str(config.image_height) + 'size'


if os.getenv('debug') is not None:
    is_debug = True if str(os.environ['debug']) == 'True' else False
else:
    is_debug = False

if os.getenv('full_depth_resnet') is not None:
    full_depth_resnet = True if str(os.environ['full_depth_resnet']) == 'True' else False

def set_random_seed(seed, deterministic=False):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not engine.cpu_only:
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def plot_grads(model, step, writer, embeddings=False):
    
    #Weights
    branch1_backbone_mean = []
    branch1_backbone_std = []
    branch1_depth_backbone_mean = []
    branch1_depth_backbone_std = []
    branch1_aspp_mean = []
    branch1_aspp_std = []
    branch1_depth_aspp_mean = []
    branch1_depth_aspp_std = []
    branch1_depth_e3_mean = []
    branch1_depth_e3_std = []
    branch1_depth_e1_mean = []
    branch1_depth_e1_std = []
    branch1_last_conv_mean = []
    branch1_last_conv_std = []
    branch1_classifier_mean = []
    branch1_classifier_std = []
    
    #Gradients
    branch1_backbone_mean_grads = []
    branch1_backbone_std_grads = []
    branch1_depth_backbone_mean_grads = []
    branch1_depth_backbone_std_grads = []
    branch1_aspp_mean_grads = []
    branch1_aspp_std_grads = []
    branch1_depth_aspp_mean_grads = []
    branch1_depth_aspp_std_grads = []
    branch1_depth_e3_mean_grads = []
    branch1_depth_e3_std_grads = []
    branch1_depth_e1_mean_grads = []
    branch1_depth_e1_std_grads = []
    branch1_last_conv_mean_grads = []
    branch1_last_conv_std_grads = []
    branch1_classifier_mean_grads = []
    branch1_classifier_std_grads = []

    for name, params in model.named_parameters():
        
        if name.startswith('branch1.backbone'):
            branch1_backbone_mean.append(params.data.mean().cpu())
            branch1_backbone_std.append(params.data.std().cpu())
            if config.depth_only == False:
                branch1_backbone_mean_grads.append(params.grad.mean().cpu())
                branch1_backbone_std_grads.append(params.grad.std().cpu())

        if name.startswith('branch1.head.aspp.map_convs') or name.startswith('branch1.head.aspp.pool_u2pl'):
            branch1_aspp_mean.append(params.data.mean().cpu())
            branch1_aspp_std.append(params.data.std().cpu())
            if config.depth_only == False:
                branch1_aspp_mean_grads.append(params.grad.mean().cpu())
                branch1_aspp_std_grads.append(params.grad.std().cpu())

        if embeddings:
            if name.startswith('branch1.head.e3_conv'):
                branch1_depth_e3_mean.append(params.data.mean().cpu())
                branch1_depth_e3_std.append(params.data.std().cpu())
                branch1_depth_e3_mean_grads.append(params.grad.mean().cpu())
                branch1_depth_e3_std_grads.append(params.grad.std().cpu())
            if name.startswith('branch1.head.e1_conv'):
                branch1_depth_e1_mean.append(params.data.mean().cpu())
                branch1_depth_e1_std.append(params.data.std().cpu())
                branch1_depth_e1_mean_grads.append(params.grad.mean().cpu())
                branch1_depth_e1_std_grads.append(params.grad.std().cpu())
        else:
            if name.startswith('branch1.depth_backbone'):
                branch1_depth_backbone_mean.append(params.data.mean().cpu())
                branch1_depth_backbone_std.append(params.data.std().cpu())
                branch1_depth_backbone_mean_grads.append(params.grad.mean().cpu())
                branch1_depth_backbone_std_grads.append(params.grad.std().cpu())
            if name.startswith('branch1.head.aspp.depth_map_convs') or name.startswith('branch1.aspp.pool_depth') or name.startswith('branch1.head.aspp.depth_red_conv'):
                branch1_depth_aspp_mean.append(params.data.mean().cpu())
                branch1_depth_aspp_std.append(params.data.std().cpu())
                branch1_depth_aspp_mean_grads.append(params.grad.mean().cpu())
                branch1_depth_aspp_std_grads.append(params.grad.std().cpu())

        if name.startswith('branch1.head.last_conv') :
            branch1_last_conv_mean.append(params.data.mean().cpu())
            branch1_last_conv_std.append(params.data.std().cpu())
            branch1_last_conv_mean_grads.append(params.grad.mean().cpu())
            branch1_last_conv_std_grads.append(params.grad.std().cpu())

        if name.startswith('branch1.classifier') :
            branch1_classifier_mean.append(params.data.mean().cpu())
            branch1_classifier_std.append(params.data.std().cpu())
            branch1_classifier_mean_grads.append(params.grad.mean().cpu())
            branch1_classifier_std_grads.append(params.grad.std().cpu())

    writer.add_histogram('Branch1_Image_Weights/Backbone_Mean', np.asarray(branch1_backbone_mean), global_step=step, bins='tensorflow')
    writer.add_histogram('Branch1_Image_Weights/Backbone_Std', np.asarray(branch1_backbone_std), global_step=step, bins='tensorflow')
    writer.add_histogram('Branch1_Image_Weights/ASPP_Mean', np.asarray(branch1_aspp_mean), global_step=step, bins='tensorflow')
    writer.add_histogram('Branch1_Image_Weights/ASPP_Std', np.asarray(branch1_aspp_std), global_step=step, bins='tensorflow')
    if config.depth_only == False:
        writer.add_histogram('Branch1_Image_Grads/Backbone_Mean_Grads', np.asarray(branch1_backbone_mean_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Image_Grads/Backbone_Std_Grads', np.asarray(branch1_backbone_std_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Image_Grads/ASPP_Mean_Grads', np.asarray(branch1_aspp_mean_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Image_Grads/ASPP_Std_Grads', np.asarray(branch1_aspp_std_grads), global_step=step, bins='tensorflow')
    
    if embeddings:
        writer.add_histogram('Branch1_Depth_Weights/E3_Mean', np.asarray(branch1_depth_e3_mean), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Depth_Weights/E3_Std', np.asarray(branch1_depth_e3_std), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Depth_Weights/E1_Mean', np.asarray(branch1_depth_e1_mean), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Depth_Weights/E1_Std', np.asarray(branch1_depth_e1_std), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Depth_Grads/E3_Mean_Grads', np.asarray(branch1_depth_e3_mean_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Depth_Grads/E3_Std_Grads', np.asarray(branch1_depth_e3_std_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Depth_Grads/E1_Mean_Grads', np.asarray(branch1_depth_e1_mean_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Depth_Grads/E1_Std_Grads', np.asarray(branch1_depth_e1_std_grads), global_step=step, bins='tensorflow')
       
    else:
        writer.add_histogram('Branch1_Depth_Weights/Depth_Backbone_Mean', np.asarray(branch1_depth_backbone_mean), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Depth_Weights/Depth_Backbone_Std', np.asarray(branch1_depth_backbone_std), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Depth_Weights/Depth_ASPP_Mean', np.asarray(branch1_depth_aspp_mean), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Depth_Weights/Depth_ASPP_Std', np.asarray(branch1_depth_aspp_std), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Depth_Grads/Depth_Backbone_Mean_Grads', np.asarray(branch1_depth_backbone_mean_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Depth_Grads/Depth_Backbone_Std_Grads', np.asarray(branch1_depth_backbone_std_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Depth_Grads/Depth_ASPP_Mean_Grads', np.asarray(branch1_depth_aspp_mean_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Depth_Grads/Depth_ASPP_Std_Grads', np.asarray(branch1_depth_aspp_std_grads), global_step=step, bins='tensorflow')

        writer.add_histogram('Branch1_Last_Conv_Weights/Last_Conv_Mean', np.asarray(branch1_last_conv_mean), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Last_Conv_Weights/Last_Conv_Std', np.asarray(branch1_last_conv_std), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Last_Conv_Grads/Last_Conv_Mean_Grads', np.asarray(branch1_last_conv_mean_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Last_Conv_Grads/Last_Conv_Std_Grads', np.asarray(branch1_last_conv_std_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Classifier_Weights/Classifier_Mean', np.asarray(branch1_classifier_mean), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Classifier_Weights/Classifier_Std', np.asarray(branch1_classifier_std), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Classifier_Grads/Classifier_Mean_Grads', np.asarray(branch1_classifier_mean_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch1_Classifier_Grads/Classifier_Std_Grads', np.asarray(branch1_classifier_std_grads), global_step=step, bins='tensorflow')

    
    #Weights
    branch2_backbone_mean = []
    branch2_backbone_std = []
    branch2_depth_backbone_mean = []
    branch2_depth_backbone_std = []
    branch2_aspp_mean = []
    branch2_aspp_std = []
    branch2_depth_aspp_mean = []
    branch2_depth_aspp_std = []
    branch2_depth_e3_mean = []
    branch2_depth_e3_std = []
    branch2_depth_e1_mean = []
    branch2_depth_e1_std = []
    branch2_last_conv_mean = []
    branch2_last_conv_std = []
    branch2_classifier_mean = []
    branch2_classifier_std = []
    
    #Gradients
    branch2_backbone_mean_grads = []
    branch2_backbone_std_grads = []
    branch2_depth_backbone_mean_grads = []
    branch2_depth_backbone_std_grads = []
    branch2_aspp_mean_grads = []
    branch2_aspp_std_grads = []
    branch2_depth_aspp_mean_grads = []
    branch2_depth_aspp_std_grads = []
    branch2_depth_e3_mean_grads = []
    branch2_depth_e3_std_grads = []
    branch2_depth_e1_mean_grads = []
    branch2_depth_e1_std_grads = []
    branch2_last_conv_mean_grads = []
    branch2_last_conv_std_grads = []
    branch2_classifier_mean_grads = []
    branch2_classifier_std_grads = []

    for name, params in model.named_parameters():
        if name.startswith('branch2.backbone'):
            branch2_backbone_mean.append(params.data.mean().cpu())
            branch2_backbone_std.append(params.data.std().cpu())
            if config.depth_only == False:
                branch2_backbone_mean_grads.append(params.grad.mean().cpu())
                branch2_backbone_std_grads.append(params.grad.std().cpu())
        
        if embeddings:
            if name.startswith('branch2.head.e3_conv'):
                branch2_depth_e3_mean.append(params.data.mean().cpu())
                branch2_depth_e3_std.append(params.data.std().cpu())
                branch2_depth_e3_mean_grads.append(params.grad.mean().cpu())
                branch2_depth_e3_std_grads.append(params.grad.std().cpu())
            if name.startswith('branch2.head.e1_conv'):
                branch2_depth_e1_mean.append(params.data.mean().cpu())
                branch2_depth_e1_std.append(params.data.std().cpu())
                branch2_depth_e1_mean_grads.append(params.grad.mean().cpu())
                branch2_depth_e1_std_grads.append(params.grad.std().cpu())
        else:
            if name.startswith('branch2.depth_backbone'):
                branch2_depth_backbone_mean.append(params.data.mean().cpu())
                branch2_depth_backbone_std.append(params.data.std().cpu())
                branch2_depth_backbone_mean_grads.append(params.grad.mean().cpu())
                branch2_depth_backbone_std_grads.append(params.grad.std().cpu())
            if name.startswith('branch2.head.aspp.depth_map_convs') or name.startswith('branch2.aspp.pool_depth') or name.startswith('branch2.head.aspp.depth_red_conv'):
                branch2_depth_aspp_mean.append(params.data.mean().cpu())
                branch2_depth_aspp_std.append(params.data.std().cpu())
                branch2_depth_aspp_mean_grads.append(params.grad.mean().cpu())
                branch2_depth_aspp_std_grads.append(params.grad.std().cpu())
        
        if name.startswith('branch2.head.aspp.map_convs') or name.startswith('branch1.head.aspp.pool_u2pl'):
            branch2_aspp_mean.append(params.data.mean().cpu())
            branch2_aspp_std.append(params.data.std().cpu())
            if config.depth_only == False:
                branch2_aspp_mean_grads.append(params.grad.mean().cpu())
                branch2_aspp_std_grads.append(params.grad.std().cpu())

        if name.startswith('branch2.head.last_conv') :
            branch2_last_conv_mean.append(params.data.mean().cpu())
            branch2_last_conv_std.append(params.data.std().cpu())
            branch2_last_conv_mean_grads.append(params.grad.mean().cpu())
            branch2_last_conv_std_grads.append(params.grad.std().cpu())

        if name.startswith('branch2.classifier') :
            branch2_classifier_mean.append(params.data.mean().cpu())
            branch2_classifier_std.append(params.data.std().cpu())
            branch2_classifier_mean_grads.append(params.grad.mean().cpu())
            branch2_classifier_std_grads.append(params.grad.std().cpu())


    writer.add_histogram('Branch2_Image_Weights/Backbone_Mean', np.asarray(branch2_backbone_mean), global_step=step, bins='tensorflow')
    writer.add_histogram('Branch2_Image_Weights/Backbone_Std', np.asarray(branch2_backbone_std), global_step=step, bins='tensorflow')
    writer.add_histogram('Branch2_Image_Weights/ASPP_Mean', np.asarray(branch2_aspp_mean), global_step=step, bins='tensorflow')
    writer.add_histogram('Branch2_Image_Weights/ASPP_Std', np.asarray(branch2_aspp_std), global_step=step, bins='tensorflow')
    if config.depth_only == False:
        writer.add_histogram('Branch2_Image_Grads/Backbone_Mean_Grads', np.asarray(branch2_backbone_mean_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Image_Grads/Backbone_Std_Grads', np.asarray(branch2_backbone_std_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Image_Grads/ASPP_Mean_Grads', np.asarray(branch2_aspp_mean_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Image_Grads/ASPP_Std_Grads', np.asarray(branch2_aspp_std_grads), global_step=step, bins='tensorflow')
    
    if embeddings:
        writer.add_histogram('Branch2_Depth_Weights/E3_Mean', np.asarray(branch2_depth_e3_mean), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Depth_Weights/E3_Std', np.asarray(branch2_depth_e3_std), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Depth_Weights/E1_Mean', np.asarray(branch2_depth_e1_mean), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Depth_Weights/E1_Std', np.asarray(branch2_depth_e1_std), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Depth_Grads/E3_Mean_Grads', np.asarray(branch2_depth_e3_mean_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Depth_Grads/E3_Std_Grads', np.asarray(branch2_depth_e3_std_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Depth_Grads/E1_Mean_Grads', np.asarray(branch2_depth_e1_mean_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Depth_Grads/E1_Std_Grads', np.asarray(branch2_depth_e1_std_grads), global_step=step, bins='tensorflow')
       
    else:
        writer.add_histogram('Branch2_Depth_Weights/Depth_Backbone_Mean', np.asarray(branch2_depth_backbone_mean), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Depth_Weights/Depth_Backbone_Std', np.asarray(branch2_depth_backbone_std), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Depth_Weights/Depth_ASPP_Mean', np.asarray(branch2_depth_aspp_mean), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Depth_Weights/Depth_ASPP_Std', np.asarray(branch2_depth_aspp_std), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Depth_Grads/Depth_Backbone_Mean_Grads', np.asarray(branch2_depth_backbone_mean_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Depth_Grads/Depth_Backbone_Std_Grads', np.asarray(branch2_depth_backbone_std_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Depth_Grads/Depth_ASPP_Mean_Grads', np.asarray(branch2_depth_aspp_mean_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Depth_Grads/Depth_ASPP_Std_Grads', np.asarray(branch2_depth_aspp_std_grads), global_step=step, bins='tensorflow')

        writer.add_histogram('Branch2_Last_Conv_Weights/Last_Conv_Mean', np.asarray(branch2_last_conv_mean), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Last_Conv_Weights/Last_Conv_Std', np.asarray(branch2_last_conv_std), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Last_Conv_Grads/Last_Conv_Mean_Grads', np.asarray(branch2_last_conv_mean_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Last_Conv_Grads/Last_Conv_Std_Grads', np.asarray(branch2_last_conv_std_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Classifier_Weights/Classifier_Mean', np.asarray(branch2_classifier_mean), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Classifier_Weights/Classifier_Std', np.asarray(branch2_classifier_std), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Classifier_Grads/Classifier_Mean_Grads', np.asarray(branch2_classifier_mean_grads), global_step=step, bins='tensorflow')
        writer.add_histogram('Branch2_Classifier_Grads/Classifier_Std_Grads', np.asarray(branch2_classifier_std_grads), global_step=step, bins='tensorflow')                  

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


def viz_image(imgs, gts, pred, step, epoch, name, step_test=None):
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
                        label, depth_image, pred_viz)
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


collate_fn = SegCollate()

if config.depthmix:
    import mask_gen_depth
    mask_generator = mask_gen_depth.BoxMaskGenerator(
        prop_range=config.depthmix_mask_prop_range,
        n_boxes=config.depthmix_boxmask_n_boxes,
        random_aspect_ratio=not config.depthmix_boxmask_fixed_aspect_ratio,
        prop_by_area=not config.depthmix_boxmask_by_size,
        within_bounds=not config.depthmix_boxmask_outside_bounds,
        invert=not config.depthmix_boxmask_no_invert)
    mask_collate_fn = SegCollate(batch_aug_fn=None)


else:
    import mask_gen
    mask_generator = mask_gen.BoxMaskGenerator(
        prop_range=config.cutmix_mask_prop_range,
        n_boxes=config.cutmix_boxmask_n_boxes,
        random_aspect_ratio=not config.cutmix_boxmask_fixed_aspect_ratio,
        prop_by_area=not config.cutmix_boxmask_by_size,
        within_bounds=not config.cutmix_boxmask_outside_bounds,
        invert=not config.cutmix_boxmask_no_invert)
    add_mask_params_to_batch = mask_gen.AddMaskParamsToBatch(
        mask_generator
    )
    mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)

# + '/{}'.format(experiment_name) + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))                 #Tensorboard log dir
tb_dir = config.tb_dir
logger = SummaryWriter(
    log_dir= config.tb_dir,
    comment=experiment_name
    )

# v3_embedder = SummaryWriter(
#     log_dir=tb_dir +
#     '_v3embedder',
#     comment=experiment_name)

path_best = osp.join(tb_dir, 'epoch-best_loss.pth')

parser = argparse.ArgumentParser()
os.environ['MASTER_PORT'] = '169711'

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True  # Changed to False due to error

    if engine.distributed:
        seed = engine.local_rank
    else:
        seed = config.seed

    set_random_seed(seed)

    # data loader + unsupervised data loader
    train_loader, train_sampler = get_train_loader(
        engine, CityScape, train_source=config.train_source, unsupervised=False, collate_fn=collate_fn)
    unsupervised_train_loader_0, unsupervised_train_sampler_0 = get_train_loader(
        engine, CityScape, train_source=config.unsup_source, unsupervised=True, collate_fn=mask_collate_fn)
    unsupervised_train_loader_1, unsupervised_train_sampler_1 = get_train_loader(
        engine, CityScape, train_source=config.unsup_source_1, unsupervised=True, collate_fn=collate_fn)


    #experiment_name = "Road_Only"
    #run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{config.batch_size}-tep{config.nepochs}-lr{config.lr}-wd{config.weight_decay}-{uuid.uuid4()}"
    #name = f"{experiment_name}_{run_id}"
    #wandb.init(project=PROJECT, name=name, tags='Road Only', entity = "alitaha")

    

    # config network and criterion
    # this is used for the min kept variable in CrossEntropyLess, basically
    # saying at least 50,000 valid targets per image (but summing them up
    # since the loss for an entire minibatch is computed at once)
    pixel_num = 5000 * config.batch_size // engine.world_size
    criterion = ProbOhemCrossEntropy2d(ignore_label=config.ignore_label, thresh=0.7, min_kept=pixel_num, use_weight=False) # NUMBER CHANGED TO 5000 from 50000 due to reduction in number of labels since only road labels valid
                                       
    criterion_cps = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.ignore_label)

    if engine.distributed and not engine.cpu_only:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d

    # WHERE WILL THE DEPTH VALUES BE APPENDED, RESNET ALREADY PRE-TRAINED WITH

    model = NetworkFullResnet(config.num_classes, criterion=criterion,  # change number of classes to free space only
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d,
                    full_depth_resnet=True,
                    depth_only=config.depth_only)  # need to change norm_layer to nn.BatchNorm2d since BatchNorm2d is derived from the furnace package and doesn't seem to work, it's only needed for syncing batches across multiple GPU, may be needed later
    init_weight(model.branch1.business_layer, nn.init.kaiming_normal_,  # to change it back to author's original, change from nn.BatchNorm2d to BatchNorm2d (which is referenced in the import statement above)
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    init_weight(model.branch2.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    #Loading image layers
    if config.load_checkpoint:
        image_state_dict = torch.load(config.checkpoint_path)
        image_layers_loaded = []
        own_state = model.state_dict()
        print(own_state['branch1.classifier.bias'].data, image_state_dict['model']['branch1.classifier.bias'].data)
        for name, param in image_state_dict['model'].items():
            if (name in own_state) and name.startswith(image_layer_tuple):#('branch1.head.depth') or name.startswith('branch2.head.last_conv'):
                param = param.data
                own_state[name].copy_(param)
                image_layers_loaded.append(name)
            else:
                continue
        #if isinstance(para
        print('Image layers loaded: ', image_layers_loaded)
        print('Should be different to verify depth classifier loaded', model.branch1.classifier.bias.data, image_state_dict['model']['branch1.classifier.bias'].data)

        print('Image Checkpoint loaded: ', config.checkpoint_path)
    else:
        print('No Image Checkpont Loaded')

    #Loading depth layers
    if config.load_depth_checkpoint:
        depth_state_dict = torch.load(config.depth_checkpoint_path)
        depth_layers_loaded = []
        own_state = model.state_dict()
        print(own_state['branch1.classifier.bias'].data, depth_state_dict['model']['branch1.classifier.bias'].data)
        for name, param in depth_state_dict['model'].items():
            if (name in own_state) and name.startswith(depth_layer_tuple): #name.startswith('branch1.head.last_conv') or name.startswith('branch2.head.last_conv'):
                param = param.data
                own_state[name].copy_(param)
                depth_layers_loaded.append(name)
            else:
                continue
        #if isinstance(param, Parameter): # backwards compatibility for serialized parameters
        print('depth layers loaded: ', depth_layers_loaded)
        print('Should be same to verify model loaded', model.branch1.classifier.bias.data, depth_state_dict['model']['branch1.classifier.bias'].data)

        print('Depth Checkpoint loaded: ', config.depth_checkpoint_path)
    else:
        print('No Depth Checkpont Loaded')

    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}

    trainval_pre = TrainValPre(config.image_mean, config.image_std, config.dimage_mean, config.dimage_std)
    test_dataset = CityScape(data_setting, 'trainval', trainval_pre)

    test_loader = data.DataLoader(test_dataset,
                                  batch_size=1,
                                  num_workers=config.num_workers,
                                  drop_last=True,
                                  shuffle=True,
                                  pin_memory=True,
                                  sampler=train_sampler)

    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr

    params_list_l = []
    # params grouped into two groups, one with weight decay and without weight
    # decay, then returned in a 2 length list and each elements in the list
    # are the parameters with the associated learning rate (this is how
    # PyTorch is designed to handle different learning rates for different
    # module parameters)
    params_list_l = group_weight(
        params_list_l,
        model.branch1.backbone,
        BatchNorm2d,
        base_lr)
    for module in model.branch1.business_layer:
        params_list_l = group_weight(params_list_l, module, BatchNorm2d,
                                     base_lr)        # head lr * 10

    optimizer_l = torch.optim.SGD(params_list_l,
                                  lr=base_lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)

    params_list_r = []
    params_list_r = group_weight(params_list_r, model.branch2.backbone,
                                 BatchNorm2d, base_lr)
    for module in model.branch2.business_layer:
        params_list_r = group_weight(params_list_r, module, BatchNorm2d,
                                     base_lr)        # head lr * 10

    optimizer_r = torch.optim.SGD(params_list_r,
                                  lr=base_lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(
        base_lr,
        config.lr_power,
        total_iteration,
        config.niters_per_epoch *
        config.warm_up_epoch)

    if engine.distributed:
        print('distributed !!')
        if not engine.cpu_only:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        model = DDP(model).to(device)  # , device_ids=[rank])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = DataParallelModel(model, device_ids=engine.devices)
        # #should I comment out since only one GPU?
        model.to(device)

    engine.register_state(
        dataloader=train_loader,
        model=model,
        optimizer_l=optimizer_l,
        optimizer_r=optimizer_r,
        loss=0)

    if engine.continue_state_object:
        engine.restore_checkpoint()     # it will change the state dict of optimizer also

    print("Number of Params", count_params(model))

    #model = load_model(model, '/media/taha_a/T7/Datasets/cityscapes/outputs/city/snapshot/snapshot/epoch-18.pth')

    step = 0
    best_miou = 0
    iu_last = 0
    mean_IU_last = 0
    mean_pixel_acc_last = 0
    loss_sup_test_last = 0
    model.train()
    print('begin train')

    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
            unsupervised_train_sampler_0.set_epoch(epoch)
            unsupervised_train_sampler_1.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

        # generate unsupervsied crops
        if config.depthmix:
            uns_crops = []
            for _ in range(config.max_samples):
                uns_crops.append(
                    generate_random_uns_crop_pos(
                        config.img_shape_h,
                        config.img_shape_w,
                        config.image_height,
                        config.image_width))
            unsupervised_train_loader_0.dataset.uns_crops = uns_crops
            unsupervised_train_loader_1.dataset.uns_crops = uns_crops

        if is_debug:
            # the tqdm function will invoke file.write(whatever) to write
            # progress of the bar and here using sys.stdout.write will write to
            # the command window
            pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = tqdm(
                range(
                    config.niters_per_epoch),
                file=sys.stdout,
                bar_format=bar_format)

        #wandb.log({"Epoch": epoch}, step=step)
        if engine.local_rank == 0:
            logger.add_scalar('Epoch', epoch, step)

        dataloader = iter(train_loader)
        # therefore the batch will instead be iterarted within each training
        # loop using dataloader.next()
        unsupervised_dataloader_0 = iter(unsupervised_train_loader_0)
        unsupervised_dataloader_1 = iter(unsupervised_train_loader_1)

        torch.autograd.set_detect_anomaly(True)

        sum_loss_sup = 0
        sum_loss_sup_r = 0
        sum_cps = 0

        ''' supervised part '''
        for idx in pbar:

            torch.cuda.empty_cache()

            optimizer_l.zero_grad()
            optimizer_r.zero_grad()
            engine.update_iteration(epoch, idx)
            start_time = time.time()

            minibatch = next(dataloader)
            unsup_minibatch_0 = next(unsupervised_dataloader_0)
            unsup_minibatch_1 = next(unsupervised_dataloader_1)

            imgs = minibatch['data']
            gts = minibatch['label']
            unsup_imgs_0 = unsup_minibatch_0['data']
            unsup_imgs_1 = unsup_minibatch_1['data']

            if config.depthmix:
                mask_params = mask_generator.generate_depth_masks(
                    imgs.shape[0], imgs.shape[2:4], imgs[:, 3:, :, :], unsup_imgs_0, unsup_imgs_1)
                mask_params = torch.from_numpy(mask_params).long()
            else:
                mask_params = unsup_minibatch_0['mask_params']

            imgs = imgs.to(device)  # .cuda(non_blocking=True)
            gts = gts.to(device)  # .cuda(non_blocking=True)
            unsup_imgs_0 = unsup_imgs_0.to(device)  # .cuda(non_blocking=True)
            unsup_imgs_1 = unsup_imgs_1.to(device)  # .cuda(non_blocking=True)
            mask_params = mask_params.to(device)  # .cuda(non_blocking=True)

            #if step==0: logger.add_graph(model.branch1, imgs[:1,:,:,:])

            # unsupervised loss on model/branch#1
            # this is a mask the size of the image with either a value of 1 or
            # 0
            batch_mix_masks = mask_params
            # augmenting the unsupervised 0 images into the 1 images -this is
            # why we have two unsup loaders so that we can get different random
            # images to augment in every iteration
            unsup_imgs_mixed = unsup_imgs_0 * \
                (1 - batch_mix_masks) + unsup_imgs_1 * batch_mix_masks

            with torch.no_grad():
                # Estimate the pseudo-label with branch#1 & supervise branch#2
                # step defines which branch we use
                _, logits_u0_tea_1 = model(unsup_imgs_0, step=1)
                _, logits_u1_tea_1 = model(unsup_imgs_1, step=1)
                logits_u0_tea_1 = logits_u0_tea_1.detach()
                logits_u1_tea_1 = logits_u1_tea_1.detach()
                # Estimate the pseudo-label with branch#2 & supervise branch#1
                _, logits_u0_tea_2 = model(unsup_imgs_0, step=2)
                _, logits_u1_tea_2 = model(unsup_imgs_1, step=2)
                logits_u0_tea_2 = logits_u0_tea_2.detach()
                logits_u1_tea_2 = logits_u1_tea_2.detach()

            # Mix teacher predictions using same mask
            # It makes no difference whether we do this with logits or probabilities as
            # the mask pixels are either 1 or 0
            logits_cons_tea_1 = logits_u0_tea_1 * \
                (1 - batch_mix_masks) + logits_u1_tea_1 * batch_mix_masks
            # getting the pseudo label since it will be the max value
            # probability
            _, ps_label_1 = torch.max(logits_cons_tea_1, dim=1)
            ps_label_1 = ps_label_1.long()
            logits_cons_tea_2 = logits_u0_tea_2 * \
                (1 - batch_mix_masks) + logits_u1_tea_2 * batch_mix_masks
            _, ps_label_2 = torch.max(logits_cons_tea_2, dim=1)
            ps_label_2 = ps_label_2.long()

            unsup_imgs_mixed.to(device)
            # Get student#1 prediction for mixed image
            _, logits_cons_stu_1 = model(unsup_imgs_mixed, step=1)
            # Get student#2 prediction for mixed image
            _, logits_cons_stu_2 = model(unsup_imgs_mixed, step=2)

            cps_loss = criterion_cps(
                logits_cons_stu_1, ps_label_2) + criterion_cps(logits_cons_stu_2, ps_label_1)
            #dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
            #cps_loss = cps_loss / engine.world_size
            cps_loss = cps_loss * config.cps_weight

            # supervised loss on both models
            _, sup_pred_l = model(imgs, step=1)
            _, sup_pred_r = model(imgs, step=2)

            loss_sup = criterion(sup_pred_l, gts)
            #dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
            loss_sup = loss_sup / engine.world_size

            loss_sup_r = criterion(sup_pred_r, gts)
            #dist.all_reduce(loss_sup_r, dist.ReduceOp.SUM)
            loss_sup_r = loss_sup_r / engine.world_size
            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            optimizer_l.param_groups[0]['lr'] = lr
            optimizer_l.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_l.param_groups)):
                optimizer_l.param_groups[i]['lr'] = lr
            optimizer_r.param_groups[0]['lr'] = lr
            optimizer_r.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_r.param_groups)):
                optimizer_r.param_groups[i]['lr'] = lr

            loss = loss_sup + loss_sup_r + cps_loss
            loss.backward()

            optimizer_l.step()
            optimizer_r.step()
            step = step + 1

            '''
            To obtain label from network (for batch size of 1 - if more than one then dimensions would shift):
            1- Permute the image or prediction first (1,2,0)
            2- torch.argmax(2) - for a batch size of 1 that is the last dimension for label - so c x h x w becomes h x w x c

            gts (label) from data loader is right size ( H x W )
            image from data loader is (C x H x W) -  reason being while image is loaded as H x W x C, it is permuted in the TrainPre Class of the dataloader
            pred from data loader is ( N x C x H x W)
            pred after permute (if extracting one sample only) is ( H x W x C )
            pred after permute and argmax(2) is ( H x W )

            *- Label doesn't need to be permuted from data loader, however either
            '''

            print_str = 'WEAK LABELS! ' if config.weak_labels else '' \
                        + 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_sup=%.2f' % loss_sup.item() \
                        + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                        + ' loss_cps=%.4f' % cps_loss.item()

            sum_loss_sup += loss_sup.item()
            sum_loss_sup_r += loss_sup_r.item()
            sum_cps += cps_loss.item()

            if engine.local_rank == 0 and step % 20 == 0:
                logger.add_scalar('train_loss_sup', loss_sup, step)
                logger.add_scalar('train_loss_sup_r', loss_sup_r, step)
                logger.add_scalar('train_loss_cps', cps_loss, step)

                if step % 500 == 0:
                    viz_image(
                        imgs,
                        gts,
                        sup_pred_l,
                        step,
                        epoch,
                        minibatch['fn'][0],
                        None)
                
                if step % 1000 == 0:
                    plot_grads(model, step, logger)

            if step % config.validate_every == 0 or (
                    is_debug and step % 10 == 0):
                all_results = []
                prec_road = []
                prec_non_road = []
                recall_road = []
                recall_non_road = []
                mean_prec = []
                mean_recall = []
                prec_recall_metrics = [
                    prec_road,
                    prec_non_road,
                    recall_road,
                    recall_non_road,
                    mean_prec,
                    mean_recall]
                names = [
                    'prec_road',
                    'prec_non_road',
                    'recall_road',
                    'recall_non_road',
                    'mean_prec',
                    'mean_recall']
                model.eval()
                loss_sup_test = 0
                step_test = 0
                v3_feats_sample = None
                v3_labels_sample = None
                feats_sample = None
                feats_labels_sample = None

                with torch.no_grad():

                    for batch_test in tqdm(
                            test_loader,
                            desc=f"Epoch: {epoch + 1}/{config.nepochs}. Loop: Validation",
                            total=len(test_loader)):

                        step_test = step_test + 1

                        imgs_test = batch_test['data'].to(device)
                        gts_test = batch_test['label'].to(device)

                        #Embedding
                        pred_test, _, v3_feats = model.branch1(imgs_test)

                        subset = 2000

                        #project embeddings
                        if False: #(step-config.embed_every) % config.embed_every == 0 or is_debug:

                            if step_test % 10 == 0:

                                _, h, w = gts_test.shape
                                v3_feats = F.interpolate(
                                    v3_feats,
                                    size=(
                                    h,
                                    w),
                                    mode='bilinear',
                                    align_corners=True)

                                _, d, v_h, v_w = v3_feats.shape
                                v3_feats = v3_feats[0,...].view(d, -1).permute(1,0)
                                v3_labels = gts_test[0,...].view(-1).unsqueeze(1)

                                v3_feats_idx = np.arange(v3_labels.shape[0])
                                v3_feats_idx = np.random.choice(v3_feats_idx, subset, replace=False)

                                # Feats not being used only V3+ Embedded since masking and then passing into model is not acurate
                                # #generate label mask
                                # non_road_mask = (gts_test > 0)
                                # road_mask = ~non_road_mask

                                # #apply mask
                                # road_only = imgs_test*road_mask.long()
                                # non_road_only = imgs_test*non_road_mask.long()

                                # #pass through model
                                # _, road_feats, _ = model(road_only)
                                # _, non_road_feats, _ = model(non_road_only)

                                # _, d, _, _ = road_feats.shape
                                # road_feats = road_feats[0,...].view(d, -1).permute(1,0)
                                # non_road_feats = non_road_feats[0,...].view(d, -1).permute(1,0)

                                # n, _ = road_feats.shape
                                # #generating labels
                                # road_labels = torch.zeros((n, 1))
                                # non_road_labels = torch.ones((n, 1))

                                # #concat feats and labels
                                # feats_combined = torch.concat((road_feats, non_road_feats), dim=0)
                                # labels_combined = torch.concat((road_labels, non_road_labels), dim=0)

                                # #random indices generated for embedding
                                # feats_idx = np.arange(labels_combined.shape[0])
                                # feats_idx = np.random.choice(feats_idx, subset, replace=False)

                                for x in range(subset):
                                    if v3_feats_sample == None:
                                        v3_feats_sample = v3_feats[0,...].unsqueeze(0)
                                        v3_labels_sample = v3_labels[0,...].unsqueeze(0)
                                        # feats_sample = feats_combined[0,...].unsqueeze(0)
                                        # feats_labels_sample = labels_combined[0,...].unsqueeze(0)
                                    else:
                                        v3_feats_sample = torch.concat((v3_feats_sample, v3_feats[v3_feats_idx[x],...].unsqueeze(0)), dim=0)
                                        v3_labels_sample = torch.concat((v3_labels_sample, v3_labels[v3_feats_idx[x],...].unsqueeze(0)), dim=0)
                                        # feats_sample = torch.concat((feats_sample, feats_combined[feats_idx[x],...].unsqueeze(0)), dim=0)
                                        # feats_labels_sample = torch.concat((feats_labels_sample, labels_combined[feats_idx[x],...].unsqueeze(0)), dim=0)

                        loss_sup_test = loss_sup_test + \
                            criterion(pred_test, gts_test)
                        pred_test_max = torch.argmax(
                            pred_test[0, :, :, :], dim=0).long().cpu().numpy()  # .permute(1, 2, 0)
                        #pred_test_max = pred_test.argmax(2).cpu().numpy()

                        hist_tmp, labeled_tmp, correct_tmp = hist_info(
                            config.num_classes, pred_test_max, gts_test[0, :, :].cpu().numpy())

                        results_dict = {
                            'hist': hist_tmp,
                            'labeled': labeled_tmp,
                            'correct': correct_tmp}
                        all_results.append(results_dict)

                        if epoch + 1 > 20:
                            if step_test % 50 == 0:
                                viz_image(
                                    imgs_test,
                                    gts_test,
                                    pred_test,
                                    step,
                                    epoch,
                                    batch_test['fn'][0],
                                    step_test)
                        elif step_test % 50 == 0:
                            viz_image(
                                imgs_test,
                                gts_test,
                                pred_test,
                                step,
                                epoch,
                                batch_test['fn'][0],
                                step_test)

                if engine.local_rank == 0:
                    iu, mean_IU, _, mean_pixel_acc, p, mean_p, r, mean_r, mean_p_no_back, mean_r_no_back = compute_metric(
                        all_results)
                    loss_sup_test = loss_sup_test / len(test_loader)

                    _ = print_pr(p, r,
                              CityScape.get_class_names(), True)

                    if mean_IU > mean_IU_last and loss_sup_test < loss_sup_test_last:
                        if os.path.exists(path_best):
                            os.remove(path_best)
                        engine.save_checkpoint(path_best)

                mean_IU_last = mean_IU
                mean_pixel_acc_last = mean_pixel_acc
                loss_sup_test_last = loss_sup_test

                print('Supervised Training Validation Set Loss', loss)
                #print(f"Validation Metrics after {step} steps: \nPixel Accuracy {pa}\nAverage Precision {ap}\nAverage Recall {ar}\nmIoU {miou}\nIoU {iou}\nF1 Score {f1}")
                _ = print_iou(iu, mean_pixel_acc,
                              CityScape.get_class_names(), True)
                logger.add_scalar('trainval_loss_sup', loss, step)
                logger.add_scalar(
                    'Val/Mean_Pixel_Accuracy', mean_pixel_acc * 100, step)
                logger.add_scalar('Val/Mean_IoU', mean_IU * 100, step)
                logger.add_scalar('Val/Mean_Prec', round(mean_p * 100, 2), step)
                logger.add_scalar('Val/Mean_Recall', round(mean_r * 100, 2), step)

                for i, n in enumerate(CityScape.get_class_names()):
                    logger.add_scalar(f'Val/IoU_{n}', iu[i] * 100, step)
                    logger.add_scalar(f'Val/Prec_{n}', round(p[i] * 100, 2), step)
                    logger.add_scalar(f'Val/Recall_{n}', round(r[i] * 100, 2), step)                 

                f1_score = (
                    2 * mean_p * mean_r) / (
                    mean_p + mean_r)
                logger.add_scalar('Val/F1 Score', round(f1_score, 2), step)

                if False: #(step-config.embed_every) % config.embed_every == 0 or is_debug:
                    #logger.add_embedding(feats_sample, feats_labels_sample, global_step=step)
                    v3_embedder.add_embedding(v3_feats_sample, v3_labels_sample, global_step=step)
                    print('embedding added at step', step)
                    del v3_feats_sample, v3_labels_sample, v3_feats, v3_labels

                if mean_IU > best_miou:
                    best_miou = mean_IU
                    best_metrics = { 
                        'miou': mean_IU*100,
                        'iou_road': iu[0]*100,
                        'iou_nonroad': iu[1]*100,
                        'accuracy': round(mean_pixel_acc*100, 2),
                        'prec_road': round(p[0]*100, 2),
                        'prec_non_road': round(p[1]*100, 2),
                        'recall_road': round(r[0]*100, 2),
                        'recall_non_road': round(r[1]*100, 2),
                        'mean_prec': round(mean_p*100, 2),
                        'mean_recall': round(mean_r*100, 2),
                        'f1_score': round(f1_score, 2)
                    }

                #The following lines clear the embeddings from the GPU which causes out of memory error (copy model to CPU and back forces clear)
                model.cpu()
                torch.cuda.empty_cache()
                model.to(device)

                model.train()
                torch.cuda.empty_cache()

            pbar.set_description(print_str, refresh=False)

            end_time = time.time()

    hparams_dict = {
        'bn_eps': config.bn_eps,
        'bn_momentum': config.bn_momentum,
        'cps_weight': config.cps_weight,
        'contrast_weight': config.contrast_weight,
        'sup_contrast_weight': config.sup_contrast_weight,
        'labeled_ratio': config.labeled_ratio,
        'batch_size': config.batch_size,
        'optimiser': str(config.optimiser),
        'lr_power': config.lr_power,
        'fig.adam': str(config.adam_betas),
        'momentum': config.momentum,
        'fig.opti': str(config.optim_params),
        'weight_d': config.weight_decay,
        'attn_lr_': config.attn_lr_factor,
        'head_lr_': config.head_lr_factor,
        'attn_hea': config.attn_heads,
        'batch_si': config.batch_size,
        'lr': config.lr,
        'image_height': config.image_height,
        'image_width': config.image_width,
        'dimage_mean': config.dimage_mean,
        'dimage_std': config.dimage_std,
        'num_classes': config.num_classes,
        'max_depth': config.max_d,
        'depth_dataset': config.depth_ckpt
    }

    logger.add_hparams(hparams_dict, best_metrics)


    if engine.distributed and (engine.local_rank == 0):
        engine.save_and_link_checkpoint(config.snapshot_dir,
                                            config.log_dir,
                                            None, epoch)
    elif not engine.distributed:
        engine.save_and_link_checkpoint(config.snapshot_dir,
                                            config.log_dir,
                                            None, epoch)
