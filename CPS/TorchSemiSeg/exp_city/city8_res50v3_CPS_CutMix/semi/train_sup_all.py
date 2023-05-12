from __future__ import division
import sys
import os
sys.path.append(os.getcwd() + '/../../..')
sys.path.append(os.getcwd() + '/..')
from custom_collate import SegCollate
from furnace.seg_opr.metric import hist_info, compute_score, recall_and_precision, recall_and_precision_all
from furnace.engine.evaluator import Evaluator
from furnace.utils.pyt_utils import load_model
from furnace.seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d, bce2d
from furnace.engine.engine import Engine
from furnace.engine.lr_policy import WarmUpPolyLR
from furnace.utils.visualize import print_iou, show_img
from furnace.utils.init_func import init_weight, group_weight
import random
import cv2
from eval import SegEvaluator
from dataloader_all import CityScape, get_train_loader, TrainValPre
from network import Network
from config import config
from PIL import Image
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import torch.distributed as dist
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

'''
VALIDATION NEEDS TO BE CHANGED TO ONLY EVERY CONFIG.VALIDATE_EVERY
MODEL PATH BEING SAVED NEEDS TO BE CHANGED BACK TO EVERY EPOCH
'''

#from furnace.seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d

PROJECT = 'CPS'
experiment_name = str(config.nepochs) + 'E_FS' + \
    str(config.labeled_ratio) + '_L' + str(config.lr) + '_NoD_'
print('File Name: No Depth - Fully Supervised')

if os.getenv('debug') is not None:
    is_debug = True if str(os.environ['debug']) == 'True' else False
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

def plot_grads(model, step, writer, embeddings=False):
    backbone_mean = []
    backbone_std = []
    depth_backbone_mean = []
    depth_backbone_std = []
    aspp_mean = []
    aspp_std = []
    depth_aspp_mean = []
    depth_aspp_std = []
    depth_e3_mean = []
    depth_e3_std = []
    depth_e1_mean = []
    depth_e1_std = []

    for name, params in model.named_parameters():
        if name.startswith('branch1.backbone'):
            backbone_mean.append(params.data.mean())
            backbone_std.append(params.data.std())
        if embeddings:

            if name.startswith('branch1.head.e3_conv'):
                depth_e3_mean.append(params.data.mean())
                depth_e3_std.append(params.data.std())
            if name.startswith('branch1.head.e1_conv'):
                depth_e1_mean.append(params.data.mean())
                depth_e1_std.append(params.data.std())
        else:
            if name.startswith('branch1.depth_backbone'):
                depth_backbone_mean.append(params.data.mean())
                depth_backbone_std.append(params.data.std())
            if name.startswith('branch1.head.aspp.depth_map_convs') or name.startswith('branch1.head.aspp.depth_downsample') or name.startswith('branch1.aspp.pool_depth'):
                depth_aspp_mean.append(params.data.mean())
                depth_aspp_std.append(params.data.std())
        if name.startswith('branch1.head.aspp.map_convs') or name.startswith('branch1.head.aspp.pool_u2pl'):
            aspp_mean.append(params.data.mean())
            aspp_std.append(params.data.std())


    writer.add_histogram('Image_Weights/Backbone_Mean', np.asarray(backbone_mean), global_step=step, bins='tensorflow')
    writer.add_histogram('Image_Weights/Backbone_Std', np.asarray(backbone_std), global_step=step, bins='tensorflow')
    writer.add_histogram('Image_Weights/ASPP_Mean', np.asarray(aspp_mean), global_step=step, bins='tensorflow')
    writer.add_histogram('Image_Weights/ASPP_Std', np.asarray(aspp_std), global_step=step, bins='tensorflow')


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

    return iu, mean_IU, _, mean_pixel_acc


def viz_image(imgs, gts, pred, step, epoch, name, step_test=None):
    image_viz = (imgs[0,
                      :,
                      :,
                      :].squeeze().cpu().numpy() * np.expand_dims(np.expand_dims(config.image_std,
                                                                                 axis=1),
                                                                  axis=2) + np.expand_dims(np.expand_dims(config.image_mean,
                                                                                                          axis=1),
                                                                                           axis=2)) * 255.0
    image_viz = image_viz.transpose(1, 2, 0)
    label = np.asarray(gts[0, :, :].squeeze().cpu(), dtype=np.uint8)
    clean = np.zeros(label.shape)
    pred_viz = torch.argmax(pred[0, :, :, :].squeeze(), dim=0).cpu()
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


collate_fn = SegCollate()

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
parser = argparse.ArgumentParser()
os.environ['MASTER_PORT'] = '169711'

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True  # Changed to False due to error

    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank

    set_random_seed(seed)

    # data loader + unsupervised data loader
    train_loader, train_sampler = get_train_loader(engine, CityScape, train_source=config.train_source,
                                                   unsupervised=False, collate_fn=collate_fn, fully_supervised=True)

    if engine.local_rank == 0:
        generate_tb_dir = config.tb_dir + '/tb'
        #engine.link_tb(tb_dir, generate_tb_dir)

    # experiment_name = "RoadOnly_FullySupervised_"
    # run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{config.batch_size}-tep{config.nepochs}-lr{config.lr}-wd{config.weight_decay}"
    # name = f"{experiment_name}_{run_id}"
    #wandb.init(project=PROJECT, name=name, tags='Road Only', entity = "alitaha")

    path_best = osp.join(tb_dir, 'epoch-best_loss.pth')

    # config network and criterion
    # this is used for the min kept variable in CrossEntropyLess, basically
    # saying at least 50,000 valid targets per image (but summing them up
    # since the loss for an entire minibatch is computed at once)
    pixel_num = 5000 * config.batch_size // engine.world_size
    criterion = ProbOhemCrossEntropy2d(ignore_label=config.ignore_label, thresh=0.7,  # NUMBER CHANGED TO 5000 from 50000 due to reduction in number of labels since only road labels valid
                                       min_kept=pixel_num, use_weight=False)

   # if engine.distributed:
    #    BatchNorm2d = SyncBatchNorm

    # WHERE WILL THE DEPTH VALUES BE APPENDED, RESNET ALREADY PRE-TRAINED WITH

    model = Network(config.num_classes, criterion=criterion,  # change number of classes to free space only
                    pretrained_model=config.pretrained_model,
                    norm_layer=nn.BatchNorm2d)  # need to change norm_layer to nn.BatchNorm2d since BatchNorm2d is derived from the furnace package and doesn't seem to work, it's only needed for syncing batches across multiple GPU, may be needed later
    init_weight(model.branch1.business_layer, nn.init.kaiming_normal_,  # to change it back to author's original, change from nn.BatchNorm2d to BatchNorm2d (which is referenced in the import statement above)
                nn.BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}

    trainval_pre = TrainValPre(config.image_mean, config.image_std)
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
        nn.BatchNorm2d,
        base_lr)
    for module in model.branch1.business_layer:
        params_list_l = group_weight(params_list_l, module, nn.BatchNorm2d,
                                     base_lr)        # head lr * 10

    optimizer_l = torch.optim.SGD(params_list_l,
                                  lr=base_lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)

    params_list_r = []
    params_list_r = group_weight(params_list_r, model.branch2.backbone,
                                 nn.BatchNorm2d, base_lr)
    for module in model.branch2.business_layer:
        params_list_r = group_weight(params_list_r, module, nn.BatchNorm2d,
                                     base_lr)        # head lr * 10

    optimizer_r = torch.optim.SGD(params_list_r,
                                  lr=base_lr,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)

    # config lr policy
    total_iteration = config.nepochs * config.fully_sup_iters
    print(total_iteration)
    lr_policy = WarmUpPolyLR(
        base_lr,
        config.lr_power,
        total_iteration,
        config.fully_sup_iters *
        config.warm_up_epoch)

    if engine.distributed:
        print('distributed !!')
        if torch.cuda.is_available():
            model.cuda()
           #model = DistributedDataParallel(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = DataParallelModel(model, device_ids=engine.devices)
        # #should I comment out since only one GPU?
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model,
                          optimizer_l=optimizer_l, optimizer_r=optimizer_r)
    if engine.continue_state_object:
        engine.restore_checkpoint()     # it will change the state dict of optimizer also

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
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

        if is_debug:
            # the tqdm function will invoke file.write(whatever) to write
            # progress of the bar and here using sys.stdout.write will write to
            # the command window
            pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = tqdm(
                range(
                    config.fully_sup_iters),
                file=sys.stdout,
                bar_format=bar_format)

        #wandb.log({"Epoch": epoch}, step=step)
        logger.add_scalar('Epoch', epoch, step)

        dataloader = iter(train_loader)


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

            imgs = minibatch['data']
            gts = minibatch['label']

            imgs = imgs.to(device)
            gts = gts.to(device)

            # supervised loss on both models
            _, sup_pred_l = model(imgs, step=1)
            _, sup_pred_r = model(imgs, step=2)

            loss_sup = criterion(sup_pred_l, gts)
            #dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
            loss_sup = loss_sup / engine.world_size

            current_idx = epoch * config.fully_sup_iters + idx
            lr = lr_policy.get_lr(current_idx)

            optimizer_l.param_groups[0]['lr'] = lr
            optimizer_l.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_l.param_groups)):
                optimizer_l.param_groups[i]['lr'] = lr

            loss = loss_sup
            loss.backward()
            optimizer_l.step()
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
                        + ' Iter{}/{}:'.format(idx + 1, config.fully_sup_iters) \
                        + ' lr=%.2e' % lr \
                        + ' loss_sup=%.2f' % loss_sup.item()  # \
            # + ' loss_sup_r=%.2f' % loss_sup_r.item() \
            #+ ' loss_cps=%.4f' % cps_loss.item()

            sum_loss_sup += loss_sup.item()

            if step % 20 == 0:
                logger.add_scalar('train_loss_sup', loss_sup, step)

                if step % 100 == 0:
                    viz_image(imgs, gts, sup_pred_l, step, epoch, minibatch['fn'][0], None)


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

                    concat_image = np.zeros((1024, 2048, 500), dtype=np.uint8)
                    concat_gt = np.zeros((1024, 2048, 500), dtype=np.uint8)

                    for batch_test in tqdm(
                            test_loader,
                            desc=f"Epoch: {epoch + 1}/{config.nepochs}. Loop: Validation",
                            total=len(test_loader)):

                        step_test = step_test + 1

                        imgs_test = batch_test['data'].to(device)
                        gts_test = batch_test['label'].to(device)
                        #pred_test = model.branch1(imgs_test)
                        #Embedding
                        pred_test, _, v3_feats = model.branch1(imgs_test)

                        subset = 2000

                        #project embeddings
                        if False: #(step-config.embed_every) % config.embed_every == 0:

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
                        
                        concat_image[:, :, (step_test-1)] = pred_test_max
                        concat_gt[:, :, (step_test-1)] = gts_test[0, :, :].cpu().numpy()

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
                    iu, mean_IU, _, mean_pixel_acc = compute_metric(
                        all_results)
                    loss_sup_test = loss_sup_test / len(test_loader)
                    
                    concat_image = concat_image.astype(np.uint8)
                    concat_gt = concat_image.astype(np.uint8)
                        
                    p, mean_p, r, mean_r = recall_and_precision_all(
                        concat_image.reshape((1024, -1)), concat_gt.reshape(1024, -1), config.num_classes)

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
                logger.add_scalar('Val/Mean_Pixel_Accuracy', mean_pixel_acc * 100, step)
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

                if False: #(step-config.embed_every) % config.embed_every == 0:
                    #logger.add_embedding(feats_sample, feats_labels_sample, global_step=step)
                    v3_embedder.add_embedding(v3_feats_sample, v3_labels_sample, global_step=step)
                    print('embedding added at step', step)

                if mean_IU > best_miou:
                    best_miou = mean_IU
                    best_metrics = { 
                        'miou': mean_IU*100,
                        'iou_road': iu[0]*100,
                        'iou_car': iu[13]*100,
                        'iou_person': iu[11]*100,
                        'iou_traffic_light': iu[6]*100,
                        'iou_vegetation': iu[8]*100,
                        'accuracy': round(mean_pixel_acc*100, 2),
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
        'num_classes': config.num_classes
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
