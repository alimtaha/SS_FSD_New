from __future__ import division
import sys
sys.path.append('../../../')
sys.path.append('../')
from custom_collate import SegCollate
import mask_gen_depth
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from furnace.seg_opr.metric import hist_info, compute_score, recall_and_precision
from furnace.engine.evaluator import Evaluator
from furnace.utils.pyt_utils import load_model
from furnace.seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d, bce2d
from furnace.engine.engine import Engine
from furnace.engine.lr_policy import WarmUpPolyLR
from furnace.utils.visualize import print_iou, show_img
from furnace.utils.init_func import init_weight, group_weight
from furnace.utils.img_utils import generate_random_uns_crop_pos
import random
import cv2
from eval_depth_concat import SegEvaluator
from dataloader_depth_concat import CityScape
from network_depth_concat import Network, count_params
from dataloader_depth_concat import get_train_loader
from config import config
from matplotlib import colors
from PIL import Image
from dataloader_depth_concat import TrainValPre
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
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
import os.path as osp


"""
import ignite.engine as i_engine
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *
"""


'''NEEED TO UPDATE VALIDATION AND EVAL FILE AND VAL PRE ETC TO INCLIDE DEPTH VALUES AND NEW FUNCTIONS'''

#from furnace.seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d

PROJECT = 'CPS'
experiment_name = str(config.nepochs) + 'E_SS' + str(config.labeled_ratio) + \
    '_L' + str(config.lr) + '_ConcatD_' + str(config.image_height) + 'size'


'''
try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

try:
    from azureml.core import Run
    azure = True
    run = Run.get_context()
except:
    azure = False
'''
if os.getenv('debug') is not None:
    is_debug = os.environ['debug']
else:
    is_debug = False


def set_random_seed(seed, deterministic=False):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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

    iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct,
                                                   labeled)
    # changed from the variable dataset to the class directly so this function
    # can now be called without first initialising the eval file
    print(len(CityScape.get_class_names()))

    '''
        if azure:
            mean_IU = np.nanmean(iu)*100
            run.log(name='Test/Val-mIoU', value=mean_IU)
        '''
    return iu, mean_IU, _, mean_pixel_acc


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
    log_dir=tb_dir +
    '/' +
    experiment_name +
    '_' +
    time.strftime(
        "%b%d_%d-%H-%M",
        time.localtime()),
    comment=experiment_name)

parser = argparse.ArgumentParser()

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

    if engine.local_rank == 0:
        generate_tb_dir = config.tb_dir + '/tb'
        engine.link_tb(tb_dir, generate_tb_dir)

    #experiment_name = "Road_Only"
    #run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{config.batch_size}-tep{config.nepochs}-lr{config.lr}-wd{config.weight_decay}-{uuid.uuid4()}"
    #name = f"{experiment_name}_{run_id}"
    #wandb.init(project=PROJECT, name=name, tags='Road Only', entity = "alitaha")

    path_best = osp.join(config.snapshot_dir, 'epoch-best_loss.pth')

    # config network and criterion
    # this is used for the min kept variable in CrossEntropyLess, basically
    # saying at least 50,000 valid targets per image (but summing them up
    # since the loss for an entire minibatch is computed at once)
    pixel_num = 5000 * config.batch_size // engine.world_size
    criterion = ProbOhemCrossEntropy2d(ignore_label=100, thresh=0.7,  # NUMBER CHANGED TO 5000 from 50000 due to reduction in number of labels since only road labels valid
                                       min_kept=pixel_num, use_weight=False)
    criterion_cps = nn.CrossEntropyLoss(reduction='mean', ignore_index=100)

    if engine.distributed and not engine.cpu_only:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d

    # WHERE WILL THE DEPTH VALUES BE APPENDED, RESNET ALREADY PRE-TRAINED WITH

    model = Network(config.num_classes, criterion=criterion,  # change number of classes to free space only
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d)  # need to change norm_layer to nn.BatchNorm2d since BatchNorm2d is derived from the furnace package and doesn't seem to work, it's only needed for syncing batches across multiple GPU, may be needed later
    init_weight(model.branch1.business_layer, nn.init.kaiming_normal_,  # to change it back to author's original, change from nn.BatchNorm2d to BatchNorm2d (which is referenced in the import statement above)
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    init_weight(model.branch2.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}

    trainval_pre = TrainValPre(
        config.image_mean,
        config.image_std,
        config.dimage_mean,
        config.dimage_std)
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

    def val_step(engine, batch):
        model.eval()
        loss_sup_test = 0
        with torch.no_grad():  # Testing
            imgs_test = batch['data'].to(device)
            gts_test = batch['label'].to(device)
            pred_test = model.branch1(imgs_test)
            loss_sup_test = loss_sup_test + criterion(pred_test, gts_test)
        return gts_test, gts_test

    """
    evaluator = i_engine.Engine(val_step)
    cm = ConfusionMatrix(num_classes=2)

    val_metrics = {
    "pixel accuracy": Accuracy(),
    "average precision": Precision(average=True),
    "average recall": Recall(average=True),
    "IoU": IoU(cm),
    "average iou": mIoU(cm),
    "F1 Score": DiceCoefficient(cm), #ignore index needs to be changed to 2 and set for all iou losses and F1
    "Loss": Loss(criterion)
    }

    for name, metric in val_metrics.items():
        metric.attach(evaluator, name)


    def log_val_results(evaluator):
        state = evaluator.run(test_loader)
        return state.metrics

    """

    #model = load_model(model, '/media/taha_a/T7/Datasets/cityscapes/outputs/city/snapshot/snapshot/epoch-18.pth')

    is_debug = False
    step = 0
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

            '''
            viz_mix = (unsup_imgs_mixed[0,:3,:,:].squeeze().numpy()*np.expand_dims(np.expand_dims(config.image_std,axis=1),axis=2)+np.expand_dims(np.expand_dims(config.image_mean,axis=1),axis=2))
            viz_mix = viz_mix.transpose(1,2,0)

            fig = plt.figure(figsize=(20, 14))
            fig.add_subplot(2, 2, 1)
            plt.imshow((unsup_imgs_0[0,:3,...].squeeze().numpy()*np.expand_dims(np.expand_dims(config.image_std,axis=1),axis=2)+np.expand_dims(np.expand_dims(config.image_mean,axis=1),axis=2)).transpose(1,2,0))
            fig.add_subplot(2, 2, 2)
            plt.imshow((unsup_imgs_1[0,:3,...].squeeze().numpy()*np.expand_dims(np.expand_dims(config.image_std,axis=1),axis=2)+np.expand_dims(np.expand_dims(config.image_mean,axis=1),axis=2)).transpose(1,2,0))
            fig.add_subplot(2, 2, 3)
            plt.imshow(viz_mix)
            fig.add_subplot(2, 2, 4)
            plt.imshow(viz_mix)
            plt.show()
            '''

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

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_sup=%.2f' % loss_sup.item() \
                        + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                        + ' loss_cps=%.4f' % cps_loss.item()

            sum_loss_sup += loss_sup.item()
            sum_loss_sup_r += loss_sup_r.item()
            sum_cps += cps_loss.item()

            if engine.local_rank == 0 and step % 20 == 0:
                #wandb.log({f"Train/Loss_Sup_R": loss_sup_r}, step=step)
                #wandb.log({f"Train/Loss_Sup_L": loss_sup}, step=step)
                #wandb.log({f"Train/Loss_CPS": cps_loss}, step=step)
                #wandb.log({f"Train/Total Loss": loss}, step=step)
                logger.add_scalar('train_loss_sup', loss_sup, step)
                logger.add_scalar('train_loss_sup_r', loss_sup_r, step)
                logger.add_scalar('train_loss_cps', cps_loss, step)

                if step % 100 == 0:
                    viz_image(
                        imgs,
                        gts,
                        sup_pred_l,
                        step,
                        epoch,
                        minibatch['fn'][0],
                        None)
                    #images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")
                    # wandb.log({"examples": images}

            if step % config.validate_every == 0 or (
                    is_debug and step % config.validate_every % 10 == 0):
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
                with torch.no_grad():
                    for batch_test in tqdm(
                            test_loader,
                            desc=f"Epoch: {epoch + 1}/{config.nepochs}. Loop: Validation",
                            total=len(test_loader)):

                        step_test = step_test + 1

                        imgs_test = batch_test['data'].to(device)
                        gts_test = batch_test['label'].to(device)
                        pred_test = model.branch1(imgs_test)
                        loss_sup_test = loss_sup_test + \
                            criterion(pred_test, gts_test)
                        pred_test_max = torch.argmax(
                            pred_test[0, :, :, :], dim=0).long().cpu().numpy()  # .permute(1, 2, 0)
                        #pred_test_max = pred_test.argmax(2).cpu().numpy()

                        hist_tmp, labeled_tmp, correct_tmp = hist_info(
                            config.num_classes, pred_test_max, gts_test[0, :, :].cpu().numpy())
                        p, mean_p, r, mean_r = recall_and_precision(
                            pred_test_max, gts_test[0, :, :].cpu().numpy(), config.num_classes)
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
                #metrics = log_val_results(evaluator)
                #pa = metrics["pixel accuracy"]
                #ap = metrics["average precision"]
                #ar = metrics["average recall"]
                #miou = metrics["average iou"]
                #f1 = metrics["F1 Score"]
                #iou = metrics["IoU"]
                #loss = metrics["Loss"]
                #loss_average = loss / len(test_loader)
                #logger.add_scalar('Val/Pixel_Accuracy', pa, step)
                #logger.add_scalar('Val/Average_Precision', ap, step)
                #logger.add_scalar('Val/Average_Recall', ar, step)
                #logger.add_scalar('Val/mIoU', miou, step)
                #logger.add_scalar('Val/F1_Score', (f1[0]+f1[1])/len(f1), step)
                print('Supervised Training Validation Set Loss', loss)
                #print(f"Validation Metrics after {step} steps: \nPixel Accuracy {pa}\nAverage Precision {ap}\nAverage Recall {ar}\nmIoU {miou}\nIoU {iou}\nF1 Score {f1}")
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
                    'Val/Precision vs Recall',
                    round(
                        prec_recall_metrics[4] *
                        100,
                        2),
                    round(
                        prec_recall_metrics[5] *
                        100,
                        2))

                model.train()

            pbar.set_description(print_str, refresh=False)

            end_time = time.time()

        # if engine.distributed and (engine.local_rank == 0):
        #logger.add_scalar('train_loss_sup', sum_loss_sup / len(pbar), epoch)
        #logger.add_scalar('train_loss_sup_r', sum_loss_sup_r / len(pbar), epoch)
        #logger.add_scalar('train_loss_cps', sum_cps / len(pbar), epoch)

        '''
        if azure and engine.local_rank == 0:
            run.log(name='Supervised Training Loss', value=sum_loss_sup / len(pbar))
            run.log(name='Supervised Training Loss right', value=sum_loss_sup_r / len(pbar))
            run.log(name='Supervised Training Loss CPS', value=sum_cps / len(pbar))
        '''

        # if '''(epoch > config.nepochs // 6) and''' (epoch %
        # config.snapshot_iter == 0) or (epoch == config.nepochs - 1):

        if engine.distributed and (engine.local_rank == 0):
            engine.save_and_link_checkpoint(config.snapshot_dir,
                                            config.log_dir,
                                            config.log_dir_link, epoch)
        elif not engine.distributed:
            engine.save_and_link_checkpoint(config.snapshot_dir,
                                            config.log_dir,
                                            config.log_dir_link, epoch)
