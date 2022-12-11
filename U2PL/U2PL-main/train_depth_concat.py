import argparse
import copy
import logging
import os
import os.path as osp
import pprint
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import colors as color
import matplotlib.cm as cm

from u2pl.dataset.augmentation import generate_unsup_data
from u2pl.dataset.builder import get_loader
from u2pl.models.model_helper import ModelBuilder
from u2pl.utils.dist_helper import setup_distributed
from u2pl.utils.loss_helper import (
    compute_contra_memobank_loss,
    compute_unsupervised_loss,
    get_criterion,
)
from u2pl.utils.lr_helper import get_optimizer, get_scheduler
from u2pl.utils.utils import (
    AverageMeter,
    get_rank,
    get_world_size,
    init_log,
    intersectionAndUnion,
    label_onehot,
    load_state,
    set_random_seed,
    uns_crops
)

from u2pl.utils.utils import (
    AverageMeter,
    check_makedirs,
    colorize,
    convert_state_dict,
    create_cityscapes_label_colormap,
    create_pascal_label_colormap,
    intersectionAndUnion,
    recall_accuracy_precision
)

parser = argparse.ArgumentParser(
    description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--port", default=None, type=int)


def viz_image(img, gts, pred, step, mode, tb_logger=None, name=None):
    image_viz = img[0,
    :3,
    :,
    :].squeeze().cpu().numpy() * np.expand_dims(np.expand_dims(std,
    axis=1),
    axis=2) + np.expand_dims(np.expand_dims(mean,
    axis=1),
     axis=2)
    image_viz = np.array(image_viz.transpose(1, 2, 0), np.uint8)
    label = np.array(gts[0, :, :], dtype=np.uint8)
    depth_image = img[0, 3:, :, :].squeeze().cpu().numpy() * dstd + dmean
    pred_viz = np.array(pred[0], np.uint8)
    pred_viz = colorize(np.uint8(pred_viz), colormap)
    label_viz = colorize(np.uint8(label), colormap)
    # set_img_color(colors, background, im1, clean, gt)
    # the pivot black bar
    final = np.array(image_viz)
    pivot = np.zeros((image_viz.shape[0], 15, 3), dtype=np.uint8)
    # stacks the images in passed through the variable argument *pd on top of
    # each other with a black bar in between
    final = np.column_stack((final, pivot))
    # Normalizing depth values
    norm = color.Normalize(0, 256)
    depth_norm = norm(np.array(depth_image))
    color_map = cm.get_cmap('magma')
    depth_color = (color_map(depth_norm) * 255)
    depth_color = np.array(depth_color, np.uint8)
    final = np.column_stack((final, depth_color[..., :3]))
    final = np.column_stack((final, pivot))
    final = np.column_stack((final, pred_viz))
    final = np.column_stack((final, pivot))
    final = np.column_stack((final, label_viz))
    if tb_logger is not None:
        # logger needs the RGB dimension as the first one
        final = final.transpose(2, 0, 1)
        if mode is "Train":
            tb_logger.add_image(
                f'{mode}/Image_Pred_GT_{step}_{name[0]}', final, step)
        else:
            tb_logger.add_image(
                f'{mode}/Image_Pred_GT_{step}_{name[0]}', final, step)
    else:
        plt.imshow(final)
        plt.show()


def main():
    global args, cfg, prototype, mean, std, dstd, dmean, colormap, i_iter
    args = parser.parse_args()
    seed = args.seed
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log("global", logging.INFO)
    logger.propagate = 0
    mean = np.array(cfg["dataset"]["mean"])
    std = np.array(cfg["dataset"]["std"])
    dmean = np.array(cfg["dataset"]["dmean"])
    dstd = np.array(cfg["dataset"]["dstd"])


experiment_name = str(
    cfg["trainer"]["epochs"]) + 'E_SS' + str(
        cfg["dataset"]["n_sup"]) + '_L' + str(
            cfg["trainer"]["optimizer"]["kwargs"]["lr"]) + '_DepthConcat' + str(
                cfg["trainer"]["unsupervised"]["apply_aug"]) + '_DataAug' + str(
                    cfg["dataset"]["train"]["crop"]["size"][0]) + '_CropSize'

 # cfg["exp_path"] = os.path.dirname(args.config)
 cfg["save_path"] = cfg["saver"]["snapshot_dir"]
  cfg["log_path"] = cfg["saver"]["log_dir"]

   if os.path.exists(cfg["save_path"]) is False:
        os.makedirs(cfg["save_path"])

    if os.path.exists(cfg["log_path"]) is False:
        os.makedirs(cfg["log_path"])

    cudnn.enabled = True  # May need to change to False
    cudnn.benchmark = True  # May need to change to False

    # rank, world_size = setup_distributed(port=args.port)
    rank = 0

    if rank == 0:
        logger.info("{}".format(pprint.pformat(cfg)))
        tb_logger = SummaryWriter(
            logdir = cfg["log_path"] + '/' + experiment_name + '_' + time.strftime("%b%d_%d-%H-%M", time.localtime()), comment = experiment_name)
    else:
        tb_logger = None

    if args.seed is not None:
        print("set random seed to", args.seed)
        set_random_seed(args.seed)

    # Create network
    model = ModelBuilder(cfg["net"])
    modules_back = [model.encoder]
    if cfg["net"].get("aux_loss", False):
        modules_head = [model.auxor, model.decoder]
    else:
        modules_head = [model.decoder]

    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            model)  # Not used since not syncing batch norm (one GPU)

    model.cuda()

    sup_loss_fn = get_criterion(cfg)

    train_loader_sup, train_loader_unsup, train_loader_aug_unsup, val_loader = get_loader(
        cfg, seed=seed)

    # Optimizer and lr decay scheduler
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    # if "pascal" in cfg["dataset"]["type"] else 1         #might need to
    # change this to 10 also since classifier head's learning rate is also
    # 10*lr on CPS
    times = 10

    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module in modules_head:
        params_list.append(
            dict(
                params=module.parameters(),
                lr=cfg_optim["kwargs"]["lr"] *
                times) )

    optimizer = get_optimizer(params_list, cfg_optim)

    # Not using DDP
    '''
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    '''

    # Teacher model
    model_teacher = ModelBuilder(cfg["net"])
    model_teacher = model_teacher.cuda()
    '''
    model_teacher = torch.nn.parallel.DistributedDataParallel(
        model_teacher,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    '''

    for p in model_teacher.parameters(
    ):  # teacher model is an EMA average of student so it doesn't need grad calculations, not optimizing through gradient descent
        p.requires_grad = False

    best_prec = 0
    last_epoch = 0

    # auto_resume > pretrain
    if cfg["saver"].get("auto_resume", False):
        lastest_model = os.path.join(cfg["save_path"], "ckpt.pth")
        if not os.path.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )
            _, _ = load_state(lastest_model, model_teacher,
                              optimizer=optimizer, key="teacher_state" )

    elif cfg["saver"].get("pretrain", False):
        load_state(cfg["saver"]["pretrain"], model, key="model_state")
        load_state(
            cfg["saver"]["pretrain"],
            model_teacher,
            key="teacher_state")

    optimizer_start = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(
        cfg_trainer,
        len(train_loader_sup),
        optimizer_start,
        start_epoch=last_epoch)

    # build class-wise memory bank
    memobank = []
    queue_ptrlis = []
    queue_size = []
    for i in range(cfg["net"]["num_classes"]):
        # len(number of classes), every element is 256
        memobank.append([torch.zeros(0, 256)])
        # halved from 30000 due to image size being reduced
        queue_size.append(15000)
        queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
    queue_size[0] = 15000  # changed from 50000 due to image size being smaller

    # build prototype
    prototype = torch.zeros(
        (
            cfg["net"]["num_classes"],
            cfg["trainer"]["contrastive"]["num_queries"],
            1,
            256,
        )
    ).cuda()

    colormap = create_cityscapes_label_colormap()

    # Start to train model
    for epoch in range(last_epoch, cfg_trainer["epochs"]):
        # Training

        print('Contrastive Loss Turned Off - line')

        i_iter = train(
            model,
            model_teacher,
            optimizer,
            lr_scheduler,
            sup_loss_fn,
            train_loader_sup,
            train_loader_unsup,
            train_loader_aug_unsup,
            epoch,
            tb_logger,
            logger,
            memobank,
            queue_ptrlis,
            queue_size,
        )

        # Validation
        if cfg_trainer["eval_on"]:
            if rank == 0:
                logger.info("start evaluation")

            if epoch < cfg["trainer"].get("sup_only_epoch", 1):
                prec = validate(
                    model,
                    val_loader,
                    epoch,
                    logger,
                    tb_logger,
                    i_iter)
            else:
                prec = validate(
                    model_teacher,
                    val_loader,
                    epoch,
                    logger,
                    tb_logger,
                    i_iter)

            if rank == 0:
                state = {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "teacher_state": model_teacher.state_dict(),
                    "best_miou": best_prec,
                }
                if prec > best_prec:
                    best_prec = prec
                    torch.save(
                        state, osp.join(cfg["save_path"], "ckpt_best.pth")
                    )

                torch.save(state, osp.join(cfg["save_path"], "ckpt.pth"))

                logger.info(
                    "\033[31m * Currently, the best val result is: {:.2f}\033[0m".format(
                        best_prec * 100
                    )
                )
                # tb_logger.add_scalar("mIoU val", prec, epoch)


def train(
    model,
    model_teacher,
    optimizer,
    lr_scheduler,
    sup_loss_fn,
    loader_l,
    loader_u,
    loader_u_aug,
    epoch,
    tb_logger,
    logger,
    memobank,
    queue_ptrlis,
    queue_size,
):
    global prototype
    ema_decay_origin = cfg["net"]["ema_decay"]

    model.train()

    uns_crops = uns_crops(
    len(loader_l) *
    cfg["dataset"]["batch_size"],
    cfg["train"]["crop"]["size"],
    cfg["val"]["crop"]["size"],
     cfg["dataset"]["ignore_label"])
    loader_u.dset.uns_crops = uns_crops
    loader_u_aug.dset.uns_crops = uns_crops

    # loader_l.sampler.set_epoch(epoch)
    # loader_u.sampler.set_epoch(epoch)
    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)
    loader_u_aug_iter = iter(loader_u_aug)
    assert len(loader_l) == len(
        loader_u
    ), f"labeled data {len(loader_l)} unlabeled data {len(loader_u)}, imbalance!"

    # rank, world_size = dist.get_rank(), dist.get_world_size()
    rank = 0
    world_size = 1

    # creates a rolling buffer of size 10 - FIFO queue?
    sup_losses = AverageMeter(10)
    uns_losses = AverageMeter(10)
    con_losses = AverageMeter(10)
    data_times = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)

    batch_end = time.time()
    for step in range(len(loader_l)):

        batch_start = time.time()
        data_times.update(batch_start - batch_end)

        i_iter = epoch * len(loader_l) + step
        lr = lr_scheduler.get_lr()
        learning_rates.update(lr[0])
        lr_scheduler.step()

        image_l, label_l, name = loader_l_iter.next()
        batch_size, h, w = label_l.size()
        image_l, label_l = image_l.cuda(), label_l.cuda()

        image_u, _, _ = loader_u_iter.next()
        image_u = image_u.cuda()
        image_u_aug, _, _ = loader_u_aug_iter.next()
        image_u_aug = image_u_aug.cuda()
        # batch_size, channels, h, w = image_u.size()
        # image_l = image_u
        # label_l = image_u[:,0,:,:]

        if epoch < cfg["trainer"].get(
            "sup_only_epoch",
            1):  # warm up epoch - currently that parameter is set at 0 so this block won't execute, will progress to else statement
            contra_flag = "none"
            # forward
            outs = model(image_l)
            pred, rep = outs["pred"], outs["rep"]
            pred = F.interpolate(
                pred, (h, w), mode="bilinear", align_corners=True)

            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = outs["aux"]
                aux = F.interpolate(
                    aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred, aux], label_l)
            else:
                sup_loss = sup_loss_fn(pred, label_l)

            model_teacher.train()
            _ = model_teacher(image_l)

            unsup_loss = 0 * rep.sum()
            contra_loss = 0 * rep.sum()
        else:
            if epoch == cfg["trainer"].get("sup_only_epoch", 1):
                # copy student parameters to teacher
                with torch.no_grad():
                    for t_params, s_params in zip(
                        model_teacher.parameters(), model.parameters()
                    ):
                        t_params.data = s_params.data

            # generate pseudo labels first
            model_teacher.eval()
            # only need probabilities from classifier head not rep head
            pred_u_teacher = model_teacher(image_u)["pred"]
            pred_u_teacher = F.interpolate(  # upsample
                pred_u_teacher, (h, w), mode="bilinear", align_corners=True
            )
            pred_u_teacher = F.softmax(pred_u_teacher, dim=1)  # softmax scores
            # predictions from softmax (both the max score as well as the
            # label/index returned for every class)
            logits_u_aug, label_u_aug = torch.max(pred_u_teacher, dim=1)

            # augmentation second loader
            # only need probabilities from classifier head not rep head
            pred_u_teacher_aug = model_teacher(image_u_aug)["pred"]
            pred_u_teacher_aug = F.interpolate(  # upsample
                pred_u_teacher_aug, (h, w), mode="bilinear", align_corners=True
            )
            pred_u_teacher_aug = F.softmax(
                pred_u_teacher_aug, dim=1)  # softmax scores
            logits_u_aug_2nd, label_u_aug_2nd = torch.max(
                pred_u_teacher_aug, dim=1)

            # apply strong data augmentation: cutout, cutmix, or classmix
            if np.random.uniform(
                    0, 1) < 0.5 and cfg["trainer"]["unsupervised"].get(
                    "apply_aug", False ):
                image_u_aug, label_u_aug, logits_u_aug = generate_unsup_data(
                    image_u,
                    label_u_aug.clone(),
                    logits_u_aug.clone(),
                    image_u_aug,
                    label_u_aug_2nd.clone(),
                    logits_u_aug_2nd.clone(),
                    ignore_label=cfg["dataset"]["ignore_label"],
                    mode=cfg["trainer"]["unsupervised"]["apply_aug"],
                )
            else:
                image_u_aug = image_u

            # viz_image(image_u_aug.cpu(), label_u_aug.cpu(), label_u_aug.cpu(), 1, 'Train', None, None)

            # forward
            num_labeled = len(image_l)  # number of pixels in labelled image
            # the supervised image and unsupervised image (pseudo labels) are
            # concatenated and fed into the model at once
            image_all = torch.cat((image_l, image_u_aug))
            outs = model(image_all)
            # the model has two heads, a prediction head and representation
            # head, so both outputs shown
            pred_all, rep_all = outs["pred"], outs["rep"]
            # separating predictions from labelled and unlabelled image
            pred_l, pred_u = pred_all[:num_labeled], pred_all[num_labeled:]
            pred_l_large = F.interpolate(
                pred_l, size=(h, w), mode="bilinear", align_corners=True
            )
            pred_u_large = F.interpolate(
                pred_u, size=(h, w), mode="bilinear", align_corners=True
            )

            # supervised loss - aux head disable for fair CPS comparison
            if "aux_loss" in cfg["net"].keys():
                aux = outs["aux"][:num_labeled]
                aux = F.interpolate(
                    aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred_l_large, aux], label_l.clone())
            else:
                sup_loss = sup_loss_fn(pred_l_large, label_l.clone())

            # teacher forward
            model_teacher.train()
            with torch.no_grad():
                out_t = model_teacher(image_all)
                # both prediction and representation head
                pred_all_teacher, rep_all_teacher = out_t["pred"], out_t["rep"]
                # softmax scores for classifier head only
                prob_all_teacher = F.softmax(pred_all_teacher, dim=1)
                prob_l_teacher, prob_u_teacher = (  # separating labelled and unlabelled image softmax scores
                    prob_all_teacher[:num_labeled],  # supervised
                    prob_all_teacher[num_labeled:],  # unsupervised
                )

                # non softmax score for unlabelled image (used for computing
                # entropy and comparing against threshold)
                pred_u_teacher = pred_all_teacher[num_labeled:]
                pred_u_large_teacher = F.interpolate(
                    pred_u_teacher, size=(
                        h, w), mode="bilinear", align_corners=True)

            # unsupervised loss
            # get (alpha_t) entropy threshold depending on current iter
            drop_percent = cfg["trainer"]["unsupervised"].get(
                "drop_percent", 100)
            percent_unreliable = (100 - drop_percent) * \
                (1 - epoch / cfg["trainer"]["epochs"])      #
            # percent unreliable gets smaller and drop percent gets larger as
            # training progresses (threshold looks at values less than entropy,
            # so drop getting smaller will mean more labels are included since
            # more will be less than the greater threshold)
            drop_percent = 100 - percent_unreliable
            unsup_loss = (
                compute_unsupervised_loss(  # this function takes in teacher pred as pseudolabels/gt, student labels and looks at drop percent to determine precentage of unreliable pixles
                    pred_u_large,
                    label_u_aug.clone(),
                    drop_percent,
                    pred_u_large_teacher.detach(),
                )
                * cfg["trainer"]["unsupervised"].get("loss_weight", 1)
            )

            # contrastive loss using unreliable pseudo labels
            contra_flag = "none"
            if cfg["trainer"].get("contrastive", False):
                cfg_contra = cfg["trainer"]["contrastive"]
                contra_flag = "{}:{}".format(
                    cfg_contra["low_rank"], cfg_contra["high_rank"]
                )
                alpha_t = cfg_contra["low_entropy_threshold"] * (
                    1 - epoch / cfg["trainer"]["epochs"]
                )

                with torch.no_grad():
                    prob = torch.softmax(pred_u_large_teacher, dim=1)
                    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

                    low_thresh = np.percentile(
                        entropy[label_u_aug != 100].cpu().numpy().flatten(), alpha_t
                    )
                    low_entropy_mask = (entropy.le(
                        low_thresh).float() * (label_u_aug != 100).bool())

                    high_thresh = np.percentile(
                        entropy[label_u_aug != 100].cpu().numpy().flatten(),
                        100 - alpha_t,
                    )
                    high_entropy_mask = (entropy.ge(
                        high_thresh).float() * (label_u_aug != 100).bool())

                    low_mask_all = torch.cat(
                        (
                            # all gt labels are obviously used since by
                            # definition gt labels have no entropy, they are
                            # true results
                            (label_l.unsqueeze(1) != 100).float(),
                            low_entropy_mask.unsqueeze(1),
                        )
                    )

                    low_mask_all = F.interpolate(  # downsamples to image size
                        low_mask_all, size=pred_all.shape[2:], mode="nearest"
                    )
                    # down sample

                    if cfg_contra.get("negative_high_entropy", True):
                        contra_flag += " high"
                        high_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 100).float(),
                                high_entropy_mask.unsqueeze(1),
                            )
                        )
                    else:
                        contra_flag += " low"
                        high_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 100).float(),
                                torch.ones(logits_u_aug.shape)
                                .float()
                                .unsqueeze(1)
                                .cuda(),
                            ),
                        )
                    high_mask_all = F.interpolate(
                        high_mask_all, size=pred_all.shape[2:], mode="nearest"
                    )  # down sample

                    # down sample and concat
                    label_l_small = F.interpolate(
                        label_onehot(label_l, cfg["net"]["num_classes"]),
                        size=pred_all.shape[2:],
                        mode="nearest",
                    )  # one hot encoded tensor downsized output
                    label_u_small = F.interpolate(
                        label_onehot(label_u_aug, cfg["net"]["num_classes"]),
                        size=pred_all.shape[2:],
                        mode="nearest",
                    )  # one hot encoded tensor downsized output

                if cfg_contra.get("binary", False):
                    contra_flag += " BCE"
                    contra_loss = compute_binary_memobank_loss(
                        rep_all,
                        torch.cat((label_l_small, label_u_small)).long(),
                        low_mask_all,
                        high_mask_all,
                        prob_all_teacher.detach(),
                        cfg_contra,
                        memobank,
                        queue_ptrlis,
                        queue_size,
                        rep_all_teacher.detach(),
                    )
                else:
                    if not cfg_contra.get("anchor_ema", False):
                        new_keys, contra_loss = compute_contra_memobank_loss(
                            rep_all,
                            label_l_small.long(),
                            label_u_small.long(),
                            prob_l_teacher.detach(),
                            prob_u_teacher.detach(),
                            low_mask_all,
                            high_mask_all,
                            cfg_contra,
                            memobank,
                            queue_ptrlis,
                            queue_size,
                            rep_all_teacher.detach(),
                        )
                    else:
                        prototype, new_keys, contra_loss = compute_contra_memobank_loss(
                            rep_all,
                            label_l_small.long(),
                            label_u_small.long(),
                            prob_l_teacher.detach(),
                            prob_u_teacher.detach(),
                            low_mask_all,
                            high_mask_all,
                            cfg_contra,
                            memobank,
                            queue_ptrlis,
                            queue_size,
                            rep_all_teacher.detach(),
                            prototype,
                        )

                # dist.all_reduce(contra_loss)

                contra_loss = (
                    contra_loss
                    # / world_size
                    * cfg["trainer"]["contrastive"].get("loss_weight", 1)
                )

            else:
                contra_loss = 0 * rep_all.sum()

        loss = sup_loss + unsup_loss + contra_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update teacher model with EMA
        if epoch >= cfg["trainer"].get("sup_only_epoch", 1):
            with torch.no_grad():
                ema_decay = min(1 -
                                1 /
                                (i_iter -
     len(loader_l) *
     cfg["trainer"].get("sup_only_epoch", 1) +
     1 ), ema_decay_origin, )
                for t_params, s_params in zip(
                    model_teacher.parameters(), model.parameters()
                ):
                    t_params.data = (ema_decay * t_params.data +
                                     (1 - ema_decay) * s_params.data )

        # gather all loss from different gpus
        reduced_sup_loss = sup_loss.clone().detach()
        # dist.all_reduce(reduced_sup_loss)
        sup_losses.update(reduced_sup_loss.item())

        reduced_uns_loss = unsup_loss.clone().detach()
        # dist.all_reduce(reduced_uns_loss)
        uns_losses.update(reduced_uns_loss.item())

        reduced_con_loss = contra_loss.clone().detach()
        # dist.all_reduce(reduced_con_loss)
        con_losses.update(reduced_con_loss.item())

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        if i_iter % 10 == 0 and rank == 0:
            logger.info(
                "[{}][{}] "
                "Iter [{}/{}]\t"
                "Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Sup {sup_loss.val:.3f} ({sup_loss.avg:.3f})\t"
                "Uns {uns_loss.val:.3f} ({uns_loss.avg:.3f})\t"
                "Con {con_loss.val:.3f} ({con_loss.avg:.3f})\t"
                "LR {lr.val:.5f}".format(
                    cfg["dataset"]["n_sup"],
                    contra_flag,
                    i_iter,
                    cfg["trainer"]["epochs"] * len(loader_l),
                    data_time=data_times,
                    batch_time=batch_times,
                    sup_loss=sup_losses,
                    uns_loss=uns_losses,
                    con_loss=con_losses,
                    lr=learning_rates,
                )
            )

            tb_logger.add_scalar("Train/lr", learning_rates.val, i_iter)
            tb_logger.add_scalar("Train/Sup_Loss", sup_losses.val, i_iter)
            tb_logger.add_scalar("Train/Uns_Loss", uns_losses.val, i_iter)
            tb_logger.add_scalar("Train/Con_Loss", con_losses.val, i_iter)
            tb_logger.add_scalar("Train/Total_Loss", loss, i_iter)

            if i_iter % 100 == 0:
                viz_image(image_l, label_l.squeeze().cpu(), pred_l_large.data.max(
                    1)[1].cpu().numpy(), i_iter, "Train", tb_logger, name)

    return i_iter


def validate(
    model,
    data_loader,
    epoch,
    logger,
    tb_logger,
    i_iter
):
    model.eval()
    # data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    # rank, world_size = dist.get_rank(), dist.get_world_size()
    rank = 0
    world_size = 1

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    acc_road = []
    acc_non_road = []
    prec_road = []
    prec_non_road = []
    recall_road = []
    recall_non_road = []
    mean_prec = []
    mean_recall = []
    mean_acc = []
    prec_recall_acc_metrics = [
        prec_road,
        prec_non_road,
        recall_road,
        recall_non_road,
        acc_road,
        acc_non_road,
        mean_prec,
        mean_recall,
        mean_acc]
    names = [
        'prec_road',
        'prec_non_road',
        'recall_road',
        'recall_non_road',
        'acc_road',
        'acc_non_road',
        'mean_prec',
        'mean_recall',
        'mean_acc']

    for step, batch in enumerate(data_loader):
        images, labels, name = batch
        images = images.cuda()
        labels = labels.long().cuda()

        with torch.no_grad():
            outs = model(images)

        # get the output produced by model_teacher
        output = outs["pred"]
        output = F.interpolate(
            output, labels.shape[1:], mode="bilinear", align_corners=True
        )
        output = output.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

        p, mean_p, r, mean_r, a, mean_a = recall_accuracy_precision(
            output, target_origin, num_classes)

        prec_recall_acc_metrics[0].append(p[0])
        prec_recall_acc_metrics[1].append(p[1])
        prec_recall_acc_metrics[2].append(r[0])
        prec_recall_acc_metrics[3].append(r[1])
        prec_recall_acc_metrics[4].append(a[0])
        prec_recall_acc_metrics[5].append(a[1])
        prec_recall_acc_metrics[6].append(mean_p)
        prec_recall_acc_metrics[7].append(mean_r)
        prec_recall_acc_metrics[8].append(mean_a)

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        # dist.all_reduce(reduced_intersection)
        # dist.all_reduce(reduced_union)
        # dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

        if epoch + 1 < 20:
            if step % 20 == 0:
                viz_image(
                    images,
                    target_origin,
                    output,
                    i_iter,
                    "Val",
                    tb_logger,
                    name)
        elif step % 10 == 0:
            viz_image(
                images,
                target_origin,
                output,
                i_iter,
                "Val",
                tb_logger,
                name)

    for idx, met in enumerate(prec_recall_acc_metrics):
        prec_recall_acc_metrics[idx] = sum(met) / len(met)

    f1_score = (2 * prec_recall_acc_metrics[6] * prec_recall_acc_metrics[7]) / (
        prec_recall_acc_metrics[6] + prec_recall_acc_metrics[7])

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)

    if rank == 0:
        for i, iou in enumerate(iou_class):
            logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
            class_label = "Road" if i == 0 else "Non_Road"
            tb_logger.add_scalar(
    f"Val/{class_label}_IoU val", iou * 100, i_iter)

        for idx, met in enumerate(prec_recall_acc_metrics):
            tb_logger.add_scalar(
    f'Val/{names[idx]}',
    round(
        met * 100,
        2),
         i_iter)
            logger.info(f"Metric:{names[idx]}: {round(met*100, 2)}")
        tb_logger.add_scalar('Val/F1 Score', round(f1_score, 2), i_iter)
        logger.info(f"Metric: F1 Score: {round(f1_score, 2)}")
        tb_logger.add_scalar(
    'Val/Precision vs Recall',
    round(
        prec_recall_acc_metrics[6] *
        100,
        2),
        round(
            prec_recall_acc_metrics[7] *
            100,
             2))

        logger.info(" * epoch {} mIoU {:.2f}".format(epoch, mIoU * 100))
        tb_logger.add_scalar("Val/mIoU val", mIoU * 100, i_iter)

    return mIoU


if __name__ == "__main__":
    main()
