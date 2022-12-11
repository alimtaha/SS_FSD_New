from __future__ import division
from functools import partial
import sys
from pathlib import Path
import importlib
sys.path.append('../../../')
sys.path.append('../')
from exp_city.city8_res50v3_CPS_CutMix.custom_collate import SegCollate
#import mask_gen_depth
#from tensorboardX import SummaryWriter
from furnace.seg_opr.metric import hist_info, compute_score, recall_and_precision
from furnace.engine.evaluator import Evaluator
from furnace.utils.pyt_utils import load_model
from furnace.seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d, BYOLContrastiveRegulizer, bce2d
from furnace.engine.engine import Engine
from furnace.engine.lr_policy import WarmUpPolyLR
from furnace.utils.visualize import print_iou, show_img
from furnace.utils.init_func import init_weight, group_weight
from furnace.utils.img_utils import generate_random_uns_crop_pos
from config import config as conzeft

import cv2
import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from matplotlib import colors
from matplotlib import pyplot as plt
from PIL import Image
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
import numpy as np
from datetime import datetime as dt
from tqdm import tqdm
import argparse
import math
import time
import random
import uuid
import os
import os.path as osp

contra = True if conzeft.mode in ['contrastive_depth_concat'] else False

#configurable import statements
package_path = conzeft.abs_dir
eval_module = importlib.import_module('eval_depth_concat', package_path)
dataloader_module = importlib.import_module('dataloader_' + 
    'depth_concat' if contra is False else conzeft.mode, package_path)
network_module = importlib.import_module('network_' + conzeft.mode, package_path)

SegEvaluator = getattr(eval_module, 'SegEvaluator')
CityScape = getattr(dataloader_module, 'CityScape')
TrainValPre = getattr(dataloader_module, 'TrainValPre')
get_train_loader = getattr(dataloader_module, 'get_train_loader')
Network = getattr(network_module, 'Network')
count_params = getattr(network_module, 'count_params')


'''NEEED TO UPDATE VALIDATION AND EVAL FILE AND VAL PRE ETC TO INCLIDE DEPTH VALUES AND NEW FUNCTIONS'''

def main(config, checkpoint_dir=None, engine=None, optim='SGD', optimising=False):

    if optimising:

        conzeft.lr = config['lr']
        conzeft.optimiser = optim
        if conzeft.optimiser != 'SGD':
            conzeft.optim_params = (config['beta1'], config['beta2'])
        else:
            conzeft.optim_params = config['momentum']
        conzeft.attn_lr_factor = config['attn_lr']
        conzeft.head_lr_factor = config['head_lr']
        conzeft.weight_decay = config['weight_decay']
        conzeft.attn_heads = 4 #config['heads']
        #conzeft.cross_att_mode = config['cross_attn_mode']

    
    notes = 'head-lr-factor' + str(conzeft.head_lr_factor) + 'attn-lr-factor' + str(conzeft.attn_lr_factor)

    hparams_dict = {
        'lr': conzeft.lr,
        'optimiser':conzeft.optimiser,
        'params': str(conzeft.optim_params),
        'decay': conzeft.weight_decay,
        'attn_lr_factor': conzeft.attn_lr_factor,
        'head_lr_factor': conzeft.head_lr_factor,
        'attn_heads': conzeft.attn_heads
    }

    print(hparams_dict)

    hparam_metric_dict = {
        'Mean_IoU': 0,
        'Mean_Pixel_Accuracy': 0,
        'Precision': 0,
        'Recall': 0,
        'IoU_Road':0,
        'IoU_NonRoad':0
    }

    optimiser = conzeft.optimiser

    if optimiser == 'Adam':
        optimiser = torch.optim.Adam
        optim_kwargs = {'betas':conzeft.optim_params}
        beta1, beta2 = conzeft.optim_params
        str_optim_params = str((round(beta1, 4), round(beta2, 4)))
    elif optimiser == 'AdamW':
        optimiser = torch.optim.AdamW
        optim_kwargs = {'betas':conzeft.optim_params}
        str_optim_params = str((round(beta1, 4), round(beta2, 4)))
    elif optimiser == 'SGD':
        optimiser = torch.optim.SGD
        optim_kwargs = {'momentum':conzeft.optim_params}
        str_optim_params = str(round(conzeft.optim_params, 4))
    else:
        raise NotImplementedError()


    PROJECT = 'CPS'
    if conzeft.weak_labels:
        experiment_name = 'weak_labels' + str(conzeft.nepochs) + 'E_SS' + str(conzeft.labeled_ratio) + \
        '_L' + str(conzeft.mode) + str(round(conzeft.lr,6)) + str(conzeft.optimiser) + str(conzeft.optim_params) +'_CrossAtt' + str(conzeft.cross_att_mode) + '_ConcatD_' + str(conzeft.image_height) + 'size_' + str(notes)
    else:
        experiment_name = str(conzeft.nepochs) + 'E_SS' + str(conzeft.labeled_ratio) + \
        '_L' + str(conzeft.mode) + str(round(conzeft.lr,6)) + str(conzeft.optimiser) + str(conzeft.optim_params) + '_CrossAtt' + str(conzeft.cross_att_mode) + '_ConcatD_' + str(conzeft.image_height) + 'size_' + str(notes)


    if os.getenv('debug') is not None:
        is_debug = True if (os.environ['debug']) == 'True' else False
    else:
        is_debug = False


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


    def compute_metric(results):
        hist = np.zeros((conzeft.num_classes, conzeft.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct,
                                                    labeled)
        # changed from the variable dataset to the class directly so this function
        # can now be called without first initialising the eval file
        print(len(CityScape.get_class_names()))

        return iu, mean_IU, _, mean_pixel_acc


    def viz_image(imgs, gts, pred, step, epoch, name, step_test=None):
        image_viz = (imgs[0,
                        :3,
                        :,
                        :].squeeze().cpu().numpy() * np.expand_dims(np.expand_dims(conzeft.image_std,
                                                                                    axis=1),
                                                                    axis=2) + np.expand_dims(np.expand_dims(conzeft.image_mean,
                                                                                                            axis=1),
                                                                                            axis=2)) * 255.0
        image_viz = image_viz.transpose(1, 2, 0)
        depth_image = imgs[0, 3:, :, :].squeeze().cpu(
        ).numpy() * conzeft.dimage_std + conzeft.dimage_mean
        label = np.asarray(gts[0, :, :].squeeze().cpu(), dtype=np.uint8)
        clean = np.zeros(label.shape)
        pred_viz = torch.argmax(pred[0, :, :, :], dim=0).cpu()
        pred_viz = np.array(pred_viz, np.uint8)
        comp_img = show_img(CityScape.get_class_colors(), conzeft.background, image_viz, clean,  # image size is 720 x 2190 x 3
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
                f'Val/Epoch_{epoch}/Val_Step_{step/conzeft.validate_every}/Image_Pred_GT_{step_test}_{name}',
                comp_img,
                step)

    collate_fn = SegCollate()

    if conzeft.depthmix:
        mask_gen_depth = importlib.import_module('mask_gen_depth', package_path)
        mask_generator = mask_gen_depth.BoxMaskGenerator(
            prop_range=conzeft.depthmix_mask_prop_range,
            n_boxes=conzeft.depthmix_boxmask_n_boxes,
            random_aspect_ratio=not conzeft.depthmix_boxmask_fixed_aspect_ratio,
            prop_by_area=not conzeft.depthmix_boxmask_by_size,
            within_bounds=not conzeft.depthmix_boxmask_outside_bounds,
            invert=not conzeft.depthmix_boxmask_no_invert)
        mask_collate_fn = SegCollate(batch_aug_fn=None)


    else:
        mask_gen = importlib.import_module('mask_gen', package_path)
        mask_generator = mask_gen.BoxMaskGenerator(
            prop_range=conzeft.cutmix_mask_prop_range,
            n_boxes=conzeft.cutmix_boxmask_n_boxes,
            random_aspect_ratio=not conzeft.cutmix_boxmask_fixed_aspect_ratio,
            prop_by_area=not conzeft.cutmix_boxmask_by_size,
            within_bounds=not conzeft.cutmix_boxmask_outside_bounds,
            invert=not conzeft.cutmix_boxmask_no_invert)

        add_mask_params_to_batch = mask_gen.AddMaskParamsToBatch(
            mask_generator
        )
        mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)

    # + '/{}'.format(experiment_name) + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))                 #Tensorboard log dir
    tb_dir = (conzeft.tb_dir 
            + '/' 
            + experiment_name
            + '_' 
            + time.strftime(
            "%b%d_%d-%H-%M",
            time.localtime()))

    logger = SummaryWriter(log_dir=tb_dir, comment=experiment_name)
    logger.add_hparams(hparam_dict=hparams_dict, metric_dict=hparam_metric_dict)



    os.environ['MASTER_PORT'] = '169711'


    eval('setattr(torch.backends.cudnn, "benchmark", True)')

    if engine.distributed:
        seed = engine.local_rank
    else:
        seed = conzeft.seed

    set_random_seed(seed)

    # data loader + unsupervised data loader
    train_loader, train_sampler = get_train_loader(
        engine, CityScape, train_source=conzeft.train_source, unsupervised=False, collate_fn=collate_fn)
    unsupervised_train_loader_0, unsupervised_train_sampler_0 = get_train_loader(
        engine, CityScape, train_source=conzeft.unsup_source, unsupervised=True, collate_fn=mask_collate_fn)
    unsupervised_train_loader_1, unsupervised_train_sampler_1 = get_train_loader(
        engine, CityScape, train_source=conzeft.unsup_source_1, unsupervised=True, collate_fn=collate_fn)

    #if engine.local_rank == 0:
    #    generate_tb_dir = conzeft.tb_dir + '/tb'
    #    #engine.link_tb(tb_dir, generate_tb_dir)

    #experiment_name = "Road_Only"
    #run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{conzeft.batch_size}-tep{conzeft.nepochs}-lr{conzeft.lr}-wd{conzeft.weight_decay}-{uuid.uuid4()}"
    #name = f"{experiment_name}_{run_id}"
    #wandb.init(project=PROJECT, name=name, tags='Road Only', entity = "alitaha")

    path_best = osp.join(tb_dir, 'epoch-best_loss.pth')

    # config network and criterion
    # this is used for the min kept variable in CrossEntropyLess, basically
    # saying at least 50,000 valid targets per image (but summing them up
    # since the loss for an entire minibatch is computed at once)
    pixel_num = 5000 * conzeft.batch_size // engine.world_size
    criterion = ProbOhemCrossEntropy2d(ignore_label=100, thresh=0.7,  # NUMBER CHANGED TO 5000 from 50000 due to reduction in number of labels since only road labels valid
                                    min_kept=pixel_num, use_weight=False)
    criterion_cps = nn.CrossEntropyLoss(reduction='mean', ignore_index=100)

    if contra:
        criterion_contrast = BYOLContrastiveRegulizer(100, 1, 1, 1)

    if engine.distributed and not engine.cpu_only:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d

    # WHERE WILL THE DEPTH VALUES BE APPENDED, RESNET ALREADY PRE-TRAINED WITH

    if conzeft.mode == 'crossattention_depth_concat':
        net_kwargs = {'heads_config': conzeft.attn_heads}
    elif conzeft.mode == 'contrastive_depth_concat':
        net_kwargs = {'ignore_param': None}
    else:
        net_kwargs = {'ignore_param': None}
    

    model = Network(conzeft.num_classes, criterion=criterion,  # change number of classes to free space only
                    pretrained_model=conzeft.pretrained_model,
                    norm_layer=BatchNorm2d, **net_kwargs)  # need to change norm_layer to nn.BatchNorm2d since BatchNorm2d is derived from the furnace package and doesn't seem to work, it's only needed for syncing batches across multiple GPU, may be needed later
    init_weight(model.branch1.business_layer, nn.init.kaiming_normal_,  # to change it back to author's original, change from nn.BatchNorm2d to BatchNorm2d (which is referenced in the import statement above)
                BatchNorm2d, conzeft.bn_eps, conzeft.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    init_weight(model.branch2.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, conzeft.bn_eps, conzeft.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    data_setting = {'img_root': conzeft.img_root_folder,
                    'gt_root': conzeft.gt_root_folder,
                    'train_source': conzeft.train_source,
                    'eval_source': conzeft.eval_source}

    trainval_pre = TrainValPre(
        conzeft.image_mean,
        conzeft.image_std,
        conzeft.dimage_mean,
        conzeft.dimage_std)
    test_dataset = CityScape(data_setting, 'trainval', trainval_pre)

    test_loader = data.DataLoader(test_dataset,
                                batch_size=1,
                                num_workers=conzeft.num_workers,
                                drop_last=True,
                                shuffle=False,
                                pin_memory=True,
                                sampler=train_sampler)

    base_lr = conzeft.lr
    if engine.distributed:
        base_lr = conzeft.lr

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
                                    base_lr*conzeft.head_lr_factor)        # head lr * 10


    params_list_r = []
    params_list_r = group_weight(params_list_r, model.branch2.backbone,
                                BatchNorm2d, base_lr)
    for module in model.branch2.business_layer:
        params_list_r = group_weight(params_list_r, module, BatchNorm2d,
                                    base_lr*conzeft.head_lr_factor)        # head lr * 10

    if conzeft.mode == 'crossattention_depth_concat':
        
        params_list_l = group_weight(
            params_list_l,
            model.branch1.cross_attention,
            BatchNorm2d,
            base_lr/conzeft.attn_lr_factor)

        params_list_r = group_weight(
            params_list_r,
            model.branch2.cross_attention,
            BatchNorm2d,
            base_lr/conzeft.attn_lr_factor)


    optimizer_l = optimiser(params_list_l,
                                lr=base_lr,
                                weight_decay=conzeft.weight_decay,
                                **optim_kwargs)

    optimizer_r = optimiser(params_list_r,
                                lr=base_lr,
                                weight_decay=conzeft.weight_decay,
                                **optim_kwargs)

    # config lr policy
    total_iteration = conzeft.nepochs * conzeft.niters_per_epoch
    lr_policy = WarmUpPolyLR(
        base_lr,
        conzeft.lr_power,
        total_iteration,
        conzeft.niters_per_epoch *
        conzeft.warm_up_epoch)

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
    iu_last = 0
    mean_IU_last = 0
    mean_pixel_acc_last = 0
    loss_sup_test_last = 0

    model.train()
    print('begin train')

    v3_feats_idx = np.load(conzeft.v3_path)
    feats_idx = np.load(conzeft.feats_path)

    for epoch in range(engine.state.epoch, conzeft.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
            unsupervised_train_sampler_0.set_epoch(epoch)
            unsupervised_train_sampler_1.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

        # generate unsupervsied crops
        if conzeft.depthmix:
            uns_crops = []
            for _ in range(conzeft.max_samples):
                uns_crops.append(
                    generate_random_uns_crop_pos(
                        conzeft.img_shape_h,
                        conzeft.img_shape_w,
                        conzeft.image_height,
                        conzeft.image_width))
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
                    conzeft.niters_per_epoch),
                file=sys.stdout,
                bar_format=bar_format)

        #wandb.log({"Epoch": epoch}, step=step)
        if engine.local_rank == 0:
            logger.add_scalar('Epoch', epoch, step)

        # the reason the dataloaders are wrapped in a iterator and then the
        # .next() method is used to load the next images is because the
        # traditional 'for batch in dataloader' can't be used since using tqdm
        dataloader = iter(train_loader)
        # therefore the batch will instead be iterarted within each training
        # loop using dataloader.next()
        unsupervised_dataloader_0 = iter(unsupervised_train_loader_0)
        unsupervised_dataloader_1 = iter(unsupervised_train_loader_1)

        torch.autograd.set_detect_anomaly(True)

        sum_loss_sup = 0
        sum_loss_sup_r = 0
        sum_cps = 0
        sum_contrast_l = 0
        sum_contrast_r = 0

        ''' supervised part '''
        for idx in pbar:

            optimizer_l.zero_grad()
            optimizer_r.zero_grad()
            engine.update_iteration(epoch, idx)
            start_time = time.time()

            minibatch = dataloader.next()
            unsup_minibatch_0 = unsupervised_dataloader_0.next()
            unsup_minibatch_1 = unsupervised_dataloader_1.next()

            imgs = minibatch['data']
            gts = minibatch['label']
            unsup_imgs_0 = unsup_minibatch_0['data']
            unsup_imgs_1 = unsup_minibatch_1['data']

            if conzeft.depthmix:
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


            if contra:
                masked_lowres = minibatch['masked']
                unsup_masked_lowres_0 = unsup_minibatch_0['masked']
                unsup_masked_lowres_1 = unsup_minibatch_1['masked']
                masked_lowres = masked_lowres.to(device)
                unsup_masked_lowres_0 = unsup_masked_lowres_0.to(device)
                unsup_masked_lowres_1 = unsup_masked_lowres_1.to(device)
                unsup_lowres_mixed = unsup_masked_lowres_0 * \
                        (1 - batch_mix_masks) + unsup_masked_lowres_1 * batch_mix_masks


            with torch.no_grad():
                # Estimate the pseudo-label with branch#1 & supervise branch#2
                # step defines which branch we use
                if contra:
                    _, logits_u0_tea_1, _, _, _  = model(unsup_imgs_0, unsup_masked_lowres_0, step=1)
                    _, logits_u1_tea_1, _, _, _ = model(unsup_imgs_1, unsup_masked_lowres_1, step=1)
                    logits_u0_tea_1 = logits_u0_tea_1.detach()
                    logits_u1_tea_1 = logits_u1_tea_1.detach()
                    # Estimate the pseudo-label with branch#2 & supervise branch#1
                    _, logits_u0_tea_2, _, _, _ = model(unsup_imgs_0, unsup_masked_lowres_0, step=2)
                    _, logits_u1_tea_2, _, _, _ = model(unsup_imgs_1, unsup_masked_lowres_1, step=2)
                    logits_u0_tea_2 = logits_u0_tea_2.detach()
                    logits_u1_tea_2 = logits_u1_tea_2.detach()
                else:
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

            if contra:
                unsup_lowres_mixed.to(device)
                _, logits_cons_stu_1, img_feat_stu_1, depth_feat_stu_1, contra_feat_stu_1 = model(unsup_imgs_mixed, unsup_lowres_mixed, step=1)
                _, logits_cons_stu_2, img_feat_stu_2,depth_feat_stu_2, contra_feat_stu_2 = model(unsup_imgs_mixed, unsup_lowres_mixed, step=2)
                contrast_loss_l = criterion_contrast(img_feat_stu_1, depth_feat_stu_2, depth=None)  * conzeft.contrast_weight
                contrast_loss_r = criterion_contrast(img_feat_stu_2, depth_feat_stu_1, depth=None)  * conzeft.contrast_weight
                if conzeft.sup_contrast:
                    _, sup_pred_l, img_feat_sup_1, depth_feat_sup_1, contra_feat_sup_1 = model(imgs, masked_lowres, step=1)
                    _, sup_pred_r, img_feat_sup_2, depth_feat_sup_2, contra_feat_sup_2 = model(imgs, masked_lowres, step=2)
                    contrast_loss_l +=  (criterion_contrast(img_feat_sup_2, depth_feat_sup_1, depth=None)  * conzeft.sup_contrast_weight)
                    contrast_loss_r += (criterion_contrast(img_feat_sup_1, depth_feat_sup_2, depth=None)  * conzeft.sup_contrast_weight)
                else:
                    _, sup_pred_l, _, _, _ = model(imgs, masked_lowres, step=1)
                    _, sup_pred_r, _, _, _ = model(imgs, masked_lowres, step=2) 
            else:
                # Get student#1 prediction for mixed image
                _, logits_cons_stu_1 = model(unsup_imgs_mixed, step=1)
                # Get student#2 prediction for mixed image
                _, logits_cons_stu_2 = model(unsup_imgs_mixed, step=2)

                _, sup_pred_l = model(imgs, step=1)
                _, sup_pred_r = model(imgs, step=2)

            cps_loss = criterion_cps(
                logits_cons_stu_1, ps_label_2) + criterion_cps(logits_cons_stu_2, ps_label_1)
            #dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
            #cps_loss = cps_loss / engine.world_size
            cps_loss = cps_loss * conzeft.cps_weight
            # supervised loss on both models
            

            loss_sup = criterion(sup_pred_l, gts)
            #dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
            loss_sup = loss_sup / engine.world_size

            loss_sup_r = criterion(sup_pred_r, gts)
            #dist.all_reduce(loss_sup_r, dist.ReduceOp.SUM)
            loss_sup_r = loss_sup_r / engine.world_size
            current_idx = epoch * conzeft.niters_per_epoch + idx
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
            if contra:
                loss = loss + contrast_loss_l + contrast_loss_r
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

            print_str = ' WEAK LABELS! ' if conzeft.weak_labels else '' \
                        + 'Epoch{}/{}'.format(epoch, conzeft.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, conzeft.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_sup=%.2f' % loss_sup.item() \
                        + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                        + ' loss_cps=%.4f' % cps_loss.item()

            sum_loss_sup += loss_sup.item()
            sum_loss_sup_r += loss_sup_r.item()
            sum_cps += cps_loss.item()

            if contra:
                print_str += \
                    ' loss_contrast_l=%.2f' % contrast_loss_l.item() \
                    + ' loss_contrast_r=%.2f' % contrast_loss_r.item() 
                sum_contrast_l += contrast_loss_l.item()
                sum_contrast_r += contrast_loss_r.item()

            if engine.local_rank == 0 and step % 20 == 0:
                logger.add_scalar('train_loss_sup', loss_sup, step)
                logger.add_scalar('train_loss_sup_r', loss_sup_r, step)
                logger.add_scalar('train_loss_cps', cps_loss, step)
                if contra:
                    logger.add_scalar('train_loss_contrast_l', contrast_loss_l, step)
                    logger.add_scalar('train_loss_contrast_r', contrast_loss_r, step)

                if step % 100 == 0:
                    viz_image(
                        imgs,
                        gts,
                        sup_pred_l,
                        step,
                        epoch,
                        minibatch['fn'][0],
                        None)

            if step % conzeft.validate_every == 0 or (
                    is_debug and step % conzeft.validate_every % 10 == 0):
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
                            desc=f"Epoch: {epoch + 1}/{conzeft.nepochs}. Loop: Validation",
                            total=len(test_loader)):

                        step_test = step_test + 1

                        imgs_test = batch_test['data'].to(device)
                        gts_test = batch_test['label'].to(device)
                        pred_test, _, v3_feats = model.branch1(imgs_test)

                        subset = 2000       #2000 * 50 images equals max of 100k points per embedding

                        #project embeddings
                        if (step-conzeft.embed_every) % conzeft.embed_every == 0:

                            if step_test % 10 == 0:

                                #take pixels for non road from top quarter of image (due to .view();s horizontal sweeping)
                                #this would mean 0-25% of h x w image size, then for road labels add 60-70% of size to indices
                                #which would mean 60/70 - 85/95 of the pixels

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

                                #generate label mask
                                non_road_mask = (gts_test > 0)
                                road_mask = ~non_road_mask

                                #apply mask
                                road_only = imgs_test*road_mask.long()
                                non_road_only = imgs_test*non_road_mask.long()

                                #pass through model
                                _, road_feats, _ = model(road_only)
                                _, non_road_feats, _ = model(non_road_only)

                                _, d, _, _ = road_feats.shape
                                road_feats = road_feats[0,...].view(d, -1).permute(1,0)
                                non_road_feats = non_road_feats[0,...].view(d, -1).permute(1,0)

                                n, _ = road_feats.shape
                                #generating labels
                                road_labels = torch.zeros((n, 1))
                                non_road_labels = torch.ones((n, 1))

                                #concat feats and labels
                                feats_combined = torch.concat((non_road_feats, road_feats), dim=0)
                                labels_combined = torch.concat((non_road_labels, road_labels), dim=0)

                                #random indices generated for embedding
                                feats_idx = np.arange(labels_combined.shape[0])
                                feats_idx = np.random.choice(feats_idx, subset, replace=False)

                                for x in range(subset):
                                    if v3_feats_sample == None:
                                        v3_feats_sample = v3_feats[0,...].unsqueeze(0)
                                        v3_labels_sample = v3_labels[0,...].unsqueeze(0)
                                        feats_sample = feats_combined[0,...].unsqueeze(0)
                                        feats_labels_sample = labels_combined[0,...].unsqueeze(0)
                                    else:
                                        v3_feats_sample = torch.concat((v3_feats_sample, v3_feats[v3_feats_idx[x],...].unsqueeze(0)), dim=0)
                                        v3_labels_sample = torch.concat((v3_labels_sample, v3_labels[v3_feats_idx[x],...].unsqueeze(0)), dim=0)
                                        feats_sample = torch.concat((feats_sample, feats_combined[feats_idx[x],...].unsqueeze(0)), dim=0)
                                        feats_labels_sample = torch.concat((feats_labels_sample, labels_combined[feats_idx[x],...].unsqueeze(0)), dim=0)

                        loss_sup_test = loss_sup_test + \
                            criterion(pred_test, gts_test)
                        pred_test_max = torch.argmax(
                            pred_test[0, :, :, :], dim=0).long().cpu().numpy()  # .permute(1, 2, 0)
                        #pred_test_max = pred_test.argmax(2).cpu().numpy()

                        hist_tmp, labeled_tmp, correct_tmp = hist_info(
                            conzeft.num_classes, pred_test_max, gts_test[0, :, :].cpu().numpy())
                        p, mean_p, r, mean_r = recall_and_precision(
                            pred_test_max, gts_test[0, :, :].cpu().numpy(), conzeft.num_classes)
                        prec_recall_metrics[0].append(p[0])
                        prec_recall_metrics[1].append(p[1])
                        prec_recall_metrics[2].append(r[0])
                        prec_recall_metrics[3].append(r[1])
                        prec_recall_metrics[4].append(mean_p)
                        prec_recall_metrics[5].append(mean_r)
                        results_dict = {
                            'hist': hist_tmp,
                            'labeled': labeled_tmp,
                            'correct': correct_tmp}
                        all_results.append(results_dict)

                        if epoch + 1 > 20:
                            if step_test % 20 == 0:
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
                    iu, mean_IU, _, mean_pixel_acc = compute_metric(
                        all_results)
                    loss_sup_test = loss_sup_test / len(test_loader)

                    if mean_IU > mean_IU_last and loss_sup_test < loss_sup_test_last:
                        if os.path.exists(path_best):
                            os.remove(path_best)
                        engine.save_checkpoint(path_best)

                mean_IU_last = mean_IU
                mean_pixel_acc_last = mean_pixel_acc
                loss_sup_test_last = loss_sup_test
                print('Supervised Training Validation Set Loss', loss)
                _ = print_iou(iu, mean_pixel_acc,
                            CityScape.get_class_names(), True)
                logger.add_scalar('trainval_loss_sup', loss, step)
                logger.add_scalar(
                    'Val/Mean_Pixel_Accuracy',
                    mean_pixel_acc * 100,
                    step)
                logger.add_scalar('Val/Mean_IoU', mean_IU * 100, step)
                logger.add_scalar('Val/IoU_Road', iu[0] * 100, step)
                logger.add_scalar('Val/IoU_NonRoad', iu[1] * 100, step)

                for i, n in enumerate(prec_recall_metrics):
                    prec_recall_metrics[i] = sum(n) / len(n)
                    logger.add_scalar(
                        f'Val/{names[i]}',
                        round(
                            prec_recall_metrics[i] *
                            100,
                            2),
                        step)
                f1_score = (
                    2 * prec_recall_metrics[4] * prec_recall_metrics[5]) / (
                    prec_recall_metrics[4] + prec_recall_metrics[5])
                logger.add_scalar('Val/F1 Score', round(f1_score, 2), step)
                logger.add_scalar(
                    'Val/Precision_vs_Recall',
                    round(
                        prec_recall_metrics[4] *
                        100,
                        2),
                    round(
                        prec_recall_metrics[5] *
                        100,
                        2))


                #Package latest metrics and get max values 
                #for add_hparam tensorboard, then log
                metric_dict_latest = {
                    'Mean_IoU': mean_IU,
                    'Mean_Pixel_Accuracy': mean_pixel_acc,
                    'Precision': round(
                        prec_recall_metrics[4] *
                        100,
                        2),
                    'Recall': round(
                        prec_recall_metrics[5] *
                        100,
                        2),
                    'IoU_Road':iu[0] * 100,
                    'IoU_NonRoad':iu[1] * 100
                }

                for k, v in hparam_metric_dict.items():
                    if v > metric_dict_latest[k]:
                        continue
                    hparam_metric_dict[k] = metric_dict_latest[k]
                logger.add_hparams(hparam_dict=hparams_dict, 
                                   metric_dict=hparam_metric_dict, 
                                   run_name=experiment_name)


                #Embeedding projector
                if (step-conzeft.embed_every) % conzeft.embed_every == 0:

                    logger.add_embedding(feats_sample, feats_labels_sample,
                                         global_step=step)
                    logger.add_embedding(v3_feats_sample, v3_labels_sample,
                                         global_step=step, tag=f'v3_{step}')

                    print('embedding added at step', step)
                
                #RayTune hyperparameter optimiser logger
                if optimising:
                    tune.report(loss=loss, miou=mean_IU, riou=iu[0],    
                                nriou=iu[1], accuracy=mean_pixel_acc,
                                precision=prec_recall_metrics[4],
                                recall=prec_recall_metrics[5])

                model.train()

            pbar.set_description(print_str, refresh=False)           

            end_time = time.time()

    # if engine.distributed and (engine.local_rank == 0):
    #logger.add_scalar('train_loss_sup', sum_loss_sup / len(pbar), epoch)
        #logger.add_scalar('train_loss_sup_r', sum_loss_sup_r / len(pbar), epoch)
        #logger.add_scalar('train_loss_cps', sum_cps / len(pbar), epoch)


if __name__ == '__main__':
   
    os.environ["PYTHONPATH"] = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/CPS/TorchSemiSeg/'

    parser = argparse.ArgumentParser()

    with Engine(custom_parser=parser) as engine:
       
        OPTIMISING = bool(os.environ['hp_optim'] in 'True')

        if OPTIMISING:

            #Optimise ADAM first
            search_adam = BayesOptSearch(metric='miou', mode='max')

            scheduler_adam = ASHAScheduler(
                metric="miou",
                mode="max",
                max_t=10,
                grace_period=4,
                reduction_factor=2
            )

            reporter_adam = CLIReporter(metric_columns=['loss', 'miou', 'accuracy'])

            config_adam = {
                'lr': tune.loguniform(1e-6, 5e-4),
                'beta1': tune.uniform(0.7, 0.99),
                'beta2': tune.loguniform(0.9, 0.999),
                'attn_lr': tune.uniform(1, 10),
                'head_lr': tune.uniform(1, 10),
                'weight_decay': tune.uniform(1e-4, 0.01),
                #'cross_attn_mode': tune.choice(['image_patch_patch', 'dual_patch_patch'
                #                                'image_token_patch', 'dual_token_patch',
                #                                'token_token'])
            }

            result_adam = tune.run(
                partial(main, engine=engine, optim='Adam', optimising=True),
                resources_per_trial={'cpu':24, 'gpu':1},
                config=config_adam,
                num_samples=30,
                search_alg=search_adam,
                scheduler=scheduler_adam,
                progress_reporter=reporter_adam,
                local_dir=conzeft.tb_dir
            )

            best_adam = result_adam.get_best_trial("miou", "max", "last")
            print(f"Best Trail Config {best_adam.config}")
            print(f"Best Trail mIoU {best_adam.last_result['miou']}")
            print(f"Best Trail mIoU {best_adam.last_result['accuracy']}")

            #Optimise SGD after ADAM
            search_sgd = BayesOptSearch(metric='miou', mode='max')

            scheduler_sgd = ASHAScheduler(
                metric="miou",
                mode="max",
                max_t=10,
                grace_period=4,
                reduction_factor=2
            )

            config_sgd = {
                'lr': tune.loguniform(1e-5, 3e-3),
                'momentum': tune.uniform(0.75, 0.99),
                'attn_lr': tune.uniform(1, 10),
                'head_lr': tune.uniform(1, 10),
                'weight_decay': tune.uniform(1e-4, 0.1),
                #'cross_attn_mode': tune.choice(['image_patch_patch', 'dual_patch_patch'
                #                                'image_token_patch', 'dual_token_patch',
                #                                'token_token'])
            }            

            reporter_sgd = CLIReporter(metric_columns=['loss', 'miou', 'accuracy'])

            result_sgd = tune.run(
                partial(main, engine=engine, optim='SGD', optimising=True),
                resources_per_trial={'cpu':20, 'gpu':1},
                config=config_sgd,
                num_samples=30,
                search_alg=search_sgd,
                scheduler=scheduler_sgd,
                progress_reporter=reporter_sgd,
                local_dir=conzeft.tb_dir
            )

            best_sgd = result_sgd.get_best_trial("miou", "max", "last")
            print(f"Best Trail Config {best_sgd.config}")
            print(f"Best Trail mIoU {best_sgd.last_result['miou']}")
            print(f"Best Trail mIoU {best_sgd.last_result['accuracy']}")

        
        else:
            main(None, None, engine=engine, optim=None, optimising=False)