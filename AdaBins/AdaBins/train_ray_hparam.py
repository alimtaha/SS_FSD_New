from functools import partial
import matplotlib
from matplotlib import pyplot as plt
import argparse
from distutils.command.sdist import sdist
import os
import sys
import uuid
from datetime import datetime as dt
from matplotlib.lines import Line2D
import random

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
#from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

#import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import model_io
import models
import utils
from dataloader import DepthDataLoader
from loss import SILogLoss, BinsChamferLoss
from utils import RunningAverage, colorize

# os.environ['WANDB_MODE'] = 'dryrun'
PROJECT = "MDE-AdaBins"
logging = True


def is_rank_zero(args):
    return args.rank == 0

def get_grad_clip(step, starting, minimum, decay_rate = 0.999):

    clip = ((starting - minimum) *
          pow(decay_rate, step // 10) +
          minimum)

    return clip

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(abs(p.grad).mean())
            max_grads.append(abs(p.grad).max())
    fig = plt.figure(figsize=(15, 10))
    '''
    print(len(layers))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=2, color="r")
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.1, lw=2, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    #plt.ylim(bottom = -0.001, top=0.04) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(False)
    plt.legend([Line2D([0], [0], color="r", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    #plt.show()
    '''
    return ave_grads, max_grads, layers


def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[:, :, :3]

    #     return img.transpose((2, 0, 1))
    return img

'''
def log_images(img, depth, pred, args, step):
    depth = colorize(depth, vmin=args.min_depth, vmax=args.max_depth)
    pred = colorize(pred, vmin=args.min_depth, vmax=args.max_depth)
    wandb.log(
        {
            "Input": [wandb.Image(img)],
            "GT": [wandb.Image(depth)],
            "Prediction": [wandb.Image(pred)]
        }, step=step)
'''

def main_worker(config, checkpoint_dir=None, gpu=0, ngpus_per_node=1, args=None):
    args.gpu = gpu

    args.lr = config['lr']
    args.w_chamfer = config['chamfer']
    args.wd = config['weight_decay']
    args.div_factor = config['div_factor']
    args.final_div_factor = config['final_div_factor']
    args.n_bins = config['n_bins']

    #Logging
    date_time = (dt.now().strftime('%m%d_%H%M'))
    experiment_name = 'AdaBins' + str(args.epochs) + 'Epochs_' + str(round(args.lr, 5)) + 'LR_' + str(args.input_height) + 'x' + str(
        args.input_width) + 'crop_' + str(args.max_depth) + 'max_depth' + str(False) + 'Log(Manual)_' + str(args.disparity) + 'Disparity_' + str(args.cpu) + '_CPU'
    print(experiment_name)
    global log_dir
    log_dir = os.path.join(args.log_dir, args.dataset, date_time)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tb_dir = os.path.join(log_dir, experiment_name)
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    
    global writer
    writer = SummaryWriter(log_dir=tb_dir)

    global result_dir
    result_dir = os.path.join(log_dir, 'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    ###################################### Load model ########################

    model = models.UnetAdaptiveBins.build(
        n_bins=args.n_bins,
        min_val=args.min_depth,
        max_val=args.max_depth,
        norm=args.norm)
     #loading kitti weights for cityscapes transfer learning
    #model = model_io.load_checkpoint(args.checkpoint_path, model)[0]
    #print('checkpoint loaded')

    ##########################################################################

    torch.cuda.empty_cache()

    # torch.manual_seed(12345)
    # np.random.seed(12345)
    # random.seed(12345)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        
        torch.cuda.set_device(args.gpu)
        # Could also use model.to(torch.device('cuda'))
        model = model.cuda(args.gpu)

    args.multigpu = False
    if args.distributed:
        # Use DDP - (not used by me)
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        # args.batch_size = 8
        args.workers = int(
            (args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print(args.gpu, args.rank, args.batch_size, args.workers)
        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[
                args.gpu],
            output_device=args.gpu,
            find_unused_parameters=True)

    elif args.gpu is None:
        # Use DP
        #args.multigpu = True
        model = model.cpu()  # if using multiple GPUs, don't specify the device type in .cuda() since then the model is passed into the nn.DataParallel method which takes care of assigning devices IDs and splitting up inputs
        #model = torch.nn.DataParallel(model)

    #print("Model\n", model)
    # print("Summary of Model\n",summary(model, (3, 352, 1216))) #print
    # summary of Model
    print("Number of Params", model_io.count_params(model))

    args.epoch = 0
    args.last_epoch = -1
    train(
        model,
        args,
        epochs=args.epochs,
        lr=args.lr,
        device=args.gpu,
        root=args.root,
        experiment_name=args.name,
        optimizer_state_dict=None)


def train(
        model,
        args,
        epochs=10,
        experiment_name="Cityscapes Depth 256depth 768bins",
        lr=0.0001,
        root=".",
        device=None,
        optimizer_state_dict=None):
    global PROJECT
    if device is None:
        device = torch.device(
            'cuda') if torch.cuda.is_available() and device is not None else torch.device('cpu')

    ###################################### Logging setup #####################
    
    print(experiment_name)

    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{args.bs}-tep{epochs}-lr{lr}-wd{args.wd}-{uuid.uuid4()}"
    name = f"{experiment_name}_{run_id}"
    should_write = ((not args.distributed) or args.rank == 0)
    should_log = should_write and logging
    if should_log:
        tags = args.tags.split(',') if args.tags != '' else None
        if args.dataset != 'nyu':
            PROJECT = PROJECT + f"-{args.dataset}"
        #wandb.init(
        #    project=PROJECT,
        #    name=name,
        #    config=args,
        #    dir=args.root,
        #    tags=tags,
        #    notes=args.notes,
        #    entity="alitaha")
        # wandb.watch(model)
    ##########################################################################

    train_loader = DepthDataLoader(args, 'train').data
    test_loader = DepthDataLoader(args, 'online_eval').data

    ###################################### losses ############################
    criterion_ueff = SILogLoss()
    criterion_bins = BinsChamferLoss() if args.chamfer else None
    ##########################################################################

    model.train()

    ###################################### Optimizer #########################
    if args.same_lr:
        print("Using same LR")
        params = model.parameters()
    else:
        print("Using diff LR")
        m = model.module if args.multigpu else model
        params = [{"params": m.get_1x_lr_params(), "lr": lr / 10},  # To train different modules in the model with different learning rates, pass the parameters and their learning rate to the optimizer in a dictionary
                  {"params": m.get_10x_lr_params(), "lr": lr}]

    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    ##########################################################################
    # some globals
    iters = len(train_loader)  # returns total number of batches
    # returns total number of all training steps (batches X epochs)
    step = args.epoch * iters
    best_loss = np.inf

    ###################################### Scheduler #########################
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        last_epoch=args.last_epoch,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor)
    if args.resume != '' and scheduler is not None:
        scheduler.step(args.epoch + 1)
    ##########################################################################

    # print max memory required by torch in GPU during training - since this
    # is before any data is passed in, this should return the size of the
    # model in theory
    print(torch.cuda.max_memory_allocated(device='cuda:0'))
    max_grads = np.zeros((1, 358))  # 358 is number of layers
    ave_grads = np.zeros((1, 358))  # 358 is number of layers
    max_grads_ep = np.zeros((1, 358))  # 358 is number of layers
    ave_grads_ep = np.zeros((1, 358))  # 358 is number of layers

    print('Model Checkpointing Disabled')

    # max_iter = len(train_loader) * epochs
    for epoch in range(args.epoch, epochs):
        ################################# Train loop ##########################
        if should_log:
            #wandb.log({"Epoch": epoch}, step=step)
            writer.add_scalar('Epoch', epoch, step)

        for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
                             total=len(train_loader)) if is_rank_zero(
                args) else enumerate(train_loader):

            optimizer.zero_grad()

            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue  # skip training iteration/step if no valid depth

            bin_edges, pred = model(img)

            '''placeholder for debugging below'
            pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)
            print(depth.shape)
            print(depth.shape[-2:])
            print(pred.shape)
            print(img.shape, "\n", img[:1,...].squeeze().cpu().shape)
            fig = plt.figure(figsize=(15, 10))
            fig.add_subplot(3, 1, 1)
            plt.title(batch['image_path'][:1])
            image_show = batch['image'][:1,...].squeeze().numpy().transpose((1,2,0))
            print(image_show)
            plt.imshow(image_show)
            fig.add_subplot(3, 1, 2)
            plt.imshow(depth[:1,...].squeeze(0).cpu().numpy().transpose((1,2,0)), cmap='magma_r')
            plt.colorbar()
            fig.add_subplot(3, 1, 3)
            plt.imshow(pred[:1,...].squeeze(0).cpu().detach().numpy().transpose((1,2,0)), cmap='magma_r')
            plt.colorbar()
            plt.show()
            '''

            mask = depth > args.min_depth
            mask_max = depth < args.max_depth
            mask = torch.logical_and(mask, mask_max)

            l_dense = criterion_ueff(
                pred, depth, mask=mask.to(
                    torch.bool), interpolate=True)

            if args.w_chamfer > 0:
                l_chamfer = criterion_bins(bin_edges, depth)
            else:
                l_chamfer = torch.Tensor([0]).to(img.device)

            loss = l_dense + args.w_chamfer * l_chamfer
            loss.backward()

            # optional - changed from 0.1 to 2 to test - CHANGE BACK?!
            clip  = get_grad_clip(step, 2, 0.01, decay_rate = 0.99)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            '''
            ave_grads_new, max_grads_new, _ = plot_grad_flow(model.named_parameters())

            if step == 0 or step % len(train_loader) == 0:
                #ave_grads[0,:] = ave_grads_new
                #max_grads[0,:] = max_grads_new
                ave_grads_ep[0,:] = ave_grads_new
                max_grads_ep[0,:] = max_grads_new
            elif  step % args.validate_every == 0:
                #ave_grads[0,:] = ave_grads_new
                #max_grads[0,:] = max_grads_new
                ave_grads_ep = np.append(ave_grads_ep, np.expand_dims(np.array(ave_grads_new), axis=0), axis=0)
                max_grads_ep = np.append(max_grads_ep, np.expand_dims(np.array(max_grads_new), axis=0), axis=0)
            else:
                #ave_grads = np.append(ave_grads, np.expand_dims(np.array(ave_grads_new), axis=0), axis=0)
                #max_grads = np.append(max_grads, np.expand_dims(np.array(max_grads_new), axis=0), axis=0)
                ave_grads_ep = np.append(ave_grads_ep, np.expand_dims(np.array(ave_grads_new), axis=0), axis=0)
                max_grads_ep = np.append(max_grads_ep, np.expand_dims(np.array(max_grads_new), axis=0), axis=0)
            '''

            if should_log and step % 20 == 0:
                writer.add_scalar(f"Train/{criterion_ueff.name}", l_dense.item(), step)
                writer.add_scalar(f"Train/{criterion_bins.name}", l_chamfer.item(), step)
                writer.add_scalar("Train/Total_Loss", loss, step)
                #wandb.log(
                #    {f"Train/{criterion_ueff.name}": l_dense.item()}, step=step)
                #wandb.log(
                #    {f"Train/{criterion_bins.name}": l_chamfer.item()}, step=step)
                #wandb.log({f"Train/Total Loss": loss}, step=step)

            step += 1
            scheduler.step()

            ###################################################################

            if should_write and step % args.validate_every == 0:

                ################################# Validation loop #############
                model.eval()

                '''
                print("Size before mean", ave_grads.shape)

                ave_grads_mean = np.mean(ave_grads, axis=0).squeeze()
                max_grads_mean = np.mean(max_grads, axis=0).squeeze()

                print("Size after mean", ave_grads_mean.shape)

                l = int(step/args.validate_every*epoch) if epoch!=0 else int(step/args.validate_every)

                data_av = [[x, y] for (x, y) in zip(range(1, len(ave_grads_mean)+1),ave_grads_mean)]
                table = wandb.Table(data=data_av, columns = ["x", "y"])
                wandb.log({f"Avg Grads {l}" : wandb.plot.line(table, "x", "y",
                title=f"Avg Grads {l}")})

                data_max = [[x, y] for (x, y) in zip(range(1, len(max_grads_mean)+1),max_grads_mean)]
                table = wandb.Table(data=data_max, columns = ["x", "y"])
                wandb.log({f"Max Grads {l}" : wandb.plot.line(table, "x", "y",
                title=f"Max Grads {l}")})
                '''
                if step % len(train_loader) == 0:

                    '''
                    ave_grads_mean_ep = np.mean(ave_grads_ep, axis=0).squeeze()
                    max_grads_mean_ep = np.mean(max_grads_ep, axis=0).squeeze()

                    l_ep = int(
                        step /
                        len(train_loader) *
                        epoch) if epoch != 0 else int(
                        step /
                        len(train_loader))

                    data_av_ep = [[x, y] for (x, y) in zip(
                        range(1, len(ave_grads_mean_ep) + 1), ave_grads_mean_ep)]
                    table = wandb.Table(data=data_av_ep, columns=["x", "y"])
                    wandb.log({f"Avg Grads Epoch{l_ep}": wandb.plot.line(
                        table, "x", "y", title=f"Avg Grads Epoch {l_ep}")})

                    data_max_ep = [[x, y] for (x, y) in zip(
                        range(1, len(max_grads_mean_ep) + 1), max_grads_mean_ep)]
                    table = wandb.Table(data=data_max_ep, columns=["x", "y"])
                    wandb.log({f"Max Grads Epoch {l_ep}": wandb.plot.line(
                        table, "x", "y", title=f"Max Grads Epoch {l_ep}")})

                    max_grads_ep = np.zeros(
                        (1, 358))  # 358 is number of layers
                    ave_grads_ep = np.zeros(
                        (1, 358))  # 358 is number of layers
                    '''

                # both logging statements replicated in commented area below
                #wandb.log({f"Gradients/Average/Step {l}": v for k, v in zip(range(1, len(ave_grads_mean)+1),ave_grads_mean)}, step=step)
                #wandb.log({f"Gradients/Max/Layer {l}": v for k, v in zip(range(1, len(max_grads_mean)+1),max_grads_mean)}, step=step)

                metrics, val_si = validate(
                    args, model, test_loader, criterion_ueff, epoch, epochs, device)

                for masks, mask_dict in metrics.items():
                    print(f"{masks} - Validated: \n {mask_dict}\n\n")
                
                if should_log:
                    
                    writer.add_scalar(f'Test/{criterion_ueff.name}', val_si.get_value(), step)
                    #wandb.log({
                    #    f"Test/{criterion_ueff.name}": val_si.get_value(),
                    #    # f"Test/{criterion_bins.name}": val_bins.get_value()
                    #}, step=step)

                    for mask, mask_dict in metrics.items():
                        if metrics[mask]['iter'] > 0:
                            for k,v in mask_dict.items():
                                if k != 'iter':
                                    writer.add_scalar(f'{mask}/{k}', v, step)
                        #wandb.log({f"Metrics/{mask_val}/{k}": v for k,
                        #           v in metrics[mask_val].items()}, step=step)
                        #wandb.log({f"Metrics/{mask_val}/{k}": v for k,
                        #           v in metrics[mask_val].items()}, step=step)
                    
                    #model_io.save_checkpoint(
                    #    model,
                    #    optimizer,
                    #    epoch,
                    #    f"{experiment_name}_{run_id}_latest.pt",
                    #    root=os.path.join(
                    #        args.log_dir,
                    #        "checkpoints"))

                #if metrics['mask_all']['abs_rel'] < best_loss and should_write:
                #    model_io.save_checkpoint(
                #        model,
                #        optimizer,
                #        epoch,
                #        f"{experiment_name}_{run_id}_best.pt",
                #        root=os.path.join(
                #            args.log_dir,
                #            "checkpoints"))
                #    best_loss = metrics['mask_all']['silog']

                # max_grads = np.zeros((1,358)) #zeroing stored gradient arrays
                # ave_grads= np.zeros((1,358)) #zeroing stored gradient arrays

                tune.report(loss=val_si.get_value(), accuracy=metrics['mask_all']['d1'])


                model.train()
                ###############################################################

    return model


def validate(
        args,
        model,
        test_loader,
        criterion_ueff,
        epoch,
        epochs,
        device='cpu'):
    with torch.no_grad():
        
        val_si = RunningAverage()
        # val_bins = RunningAverage()
        
        #metrics = utils.RunningAverageDict()
        
        masks = ['mask_20', 'mask_80', 'mask_20_80', 'mask_80_256', 'mask_all']
        metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog', 'iter']

        result_metrics = {}

        for mask in masks:
            result_metrics[mask] = {}
            for metric in metric_name:
                result_metrics[mask][metric] = 0.0

        for batch in tqdm(test_loader,desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation") if is_rank_zero(args) else test_loader:
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            filename = batch['image_path'][0].split('/')[-1]
            
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            bins, pred = model(img)

            mask = depth > args.min_depth
            mask_max = depth < args.max_depth
            mask = torch.logical_and(mask, mask_max)

            l_dense = criterion_ueff(
                pred, depth, mask=mask.to(
                    torch.bool), interpolate=True)
            
            val_si.append(l_dense.item())

            pred = nn.functional.interpolate(
                pred, depth.shape[-2:], mode='bilinear', align_corners=True)

            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth_eval] = args.min_depth_eval
            pred[pred > args.max_depth_eval] = args.max_depth_eval
            pred[np.isinf(pred)] = args.max_depth_eval
            pred[np.isnan(pred)] = args.min_depth_eval

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(
                gt_depth > args.min_depth_eval,
                gt_depth < args.max_depth_eval)

            #save validation outputs
            if args.save_result:

                if save_path.split('.')[-1] == 'jpg':
                    save_path = save_path.replace('jpg', 'png')

                save_path = os.path.join(result_dir, f'epoch_{epoch}/')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path = os.path.join(save_path, filename)

                if args.dataset == 'kitti' or args.dataset == 'cityscapes':
                    pred_d_numpy = pred * 256.0
                    cv2.imwrite(
                    save_path, pred_d_numpy.astype(
                        np.uint16), [
                        cv2.IMWRITE_PNG_COMPRESSION, 0])

            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.ones(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), 
                              int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti' or args.dataset == 'cityscapes':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                                  int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    elif args.dataset == 'nyu':
                        eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)
            #metrics.update(utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))
            metrics = utils.compute_errors(
                gt_depth[valid_mask], pred[valid_mask])

            for mask in result_metrics.keys():
                if mask in metrics.keys():
                    for key in result_metrics[mask].keys():
                        result_metrics[mask][key] += metrics[mask][key]

        for mask in result_metrics.keys():
            if result_metrics[mask]['iter'] > 0:
                for key in result_metrics[mask].keys():
                    if key != 'iter':
                        result_metrics[mask][key] /= result_metrics[mask]['iter']


        return result_metrics, val_si


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(
        description='Training script. Default values of all arguments are recommended for reproducibility',
        fromfile_prefix_chars='@',
        conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--epochs', default=15, type=int,
                        help='number of total epochs to run')
    parser.add_argument(
        '--n-bins',
        '--n_bins',
        default=80,
        type=int,
        help='number of bins/buckets to divide depth range into')
    parser.add_argument(
        '--lr',
        '--learning-rate',
        default=0.000357,
        type=float,
        help='max learning rate')
    parser.add_argument(
        '--wd',
        '--weight-decay',
        default=0.1,
        type=float,
        help='weight decay')
    parser.add_argument(
        '--w_chamfer',
        '--w-chamfer',
        default=0.1,
        type=float,
        help="weight value for chamfer loss")
    parser.add_argument(
        '--div-factor',
        '--div_factor',
        default=100,
        type=float,
        help="Initial div factor for lr")
    parser.add_argument(
        '--final-div-factor',
        '--final_div_factor',
        default=100,
        type=float,
        help="final div factor for lr")
    parser.add_argument('--bs', default=16, type=int, help='batch size')
    parser.add_argument(
        '--validate-every',
        '--validate_every',
        default=2000,
        type=int,
        help='validation period')
    parser.add_argument(
        '--gpu',
        default=None,
        type=int,
        help='Which gpu to use')
    parser.add_argument("--name", default="UnetAdaptiveBins")
    parser.add_argument(
        "--norm",
        default="linear",
        type=str,
        help="Type of norm/competition for bin-widths",
        choices=[
            'linear',
            'softmax',
            'sigmoid'])
    parser.add_argument(
        "--same-lr",
        '--same_lr',
        default=False,
        action="store_true",
        help="Use same LR for all param groups")
    # Changed default from True to False since not using multiple GPUs
    parser.add_argument(
        "--distributed",
        default=False,
        action="store_true",
        help="Use DDP if set")
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")
    parser.add_argument(
        "--resume",
        default='',
        type=str,
        help="Resume from checkpoint")
    parser.add_argument(
        '--checkpoint_path',
        '--checkpoint-path',
        type=str,
        required=False,
        help="checkpoint file to use for prediction")
    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--tags", default='sweep', type=str, help="Wandb tags")
    parser.add_argument("--workers", default=11, type=int,
                        help="Number of workers for data loading")
    parser.add_argument(
        "--dataset",
        default='nyu',
        type=str,
        help="Dataset to train on")
    parser.add_argument(
        "--data_path",
        default='../dataset/nyu/sync/',
        type=str,
        help="path to dataset")
    parser.add_argument(
        "--gt_path",
        default='../dataset/nyu/sync/',
        type=str,
        help="path to dataset")
    parser.add_argument(
        '--filenames_file',
        default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
        type=str,
        help='path to the filenames text file')
    parser.add_argument(
        '--log_dir',
        default="./logs",
        type=str,
        help='path to the results and tensorboard summary files')
    parser.add_argument(
        '--camera_path',
        default='/home/extraspace/Datasets/Datasets/cityscapes/camera/',
        type=str,
        help='path to the results and tensorboard summary files')
    parser.add_argument(
        '--disparity_path',
        default='/home/extraspace/Datasets/Datasets/cityscapes/disparity/',
        type=str,
        help='path to the results and tensorboard summary files')
    parser.add_argument(
        '--input_height',
        type=int,
        help='input height',
        default=416)
    parser.add_argument(
        '--input_width',
        type=int,
        help='input width',
        default=544)
    parser.add_argument(
        '--max_depth',
        type=float,
        help='maximum depth in estimation',
        default=10)
    parser.add_argument(
        '--min_depth',
        type=float,
        help='minimum depth in estimation',
        default=1e-3)
    parser.add_argument(
        '--do_random_rotate',
        default=True,
        help='if set, will perform random rotation for augmentation',
        action='store_true')
    parser.add_argument(
        '--degree',
        type=float,
        help='random rotation maximum degree',
        default=2.5)
    parser.add_argument(
        '--do_kb_crop',
        help='if set, crop input images as kitti benchmark images',
        action='store_true')
    parser.add_argument(
        '--use_right',
        help='if set, will randomly use right images when train on KITTI',
        action='store_true')
    parser.add_argument(
        '--save_result',
        help='if set, will save images from validation runs',
        action='store_true')
    parser.add_argument(
        '--logarithmic',
        help='if set, will use logarithmic transform on depth values during training for loss function',
        action='store_true')
    parser.add_argument(
        '--disparity',
        help='if set, will use disparity maps instead of depth maps',
        action='store_true')
    parser.add_argument(
        '--data_path_eval',
        default="../dataset/nyu/official_splits/test/",
        type=str,
        help='path to the data for online evaluation')
    parser.add_argument(
        '--gt_path_eval',
        default="../dataset/nyu/official_splits/test/",
        type=str,
        help='path to the groundtruth data for online evaluation')
    parser.add_argument(
        '--filenames_file_eval',
        default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
        type=str,
        help='path to the filenames text file for online evaluation')
    parser.add_argument(
        '--min_depth_eval',
        type=float,
        help='minimum depth for evaluation',
        default=1e-3)
    parser.add_argument(
        '--max_depth_eval',
        type=float,
        help='maximum depth for evaluation',
        default=10)
    parser.add_argument(
        '--eigen_crop',
        default=True,
        help='if set, crops according to Eigen NIPS14',
        action='store_true')
    parser.add_argument(
        '--garg_crop',
        help='if set, crops according to Garg  ECCV16',
        action='store_true')
    parser.add_argument(
        '--cpu',
        help='if set, uses CPU only',
        action='store_true')

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    args.batch_size = args.bs
    args.num_threads = args.workers
    args.mode = 'train'
    args.chamfer = args.w_chamfer > 0
    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace(
            '[', '').replace(']', '')
        nodes = node_str.split(',')

        args.world_size = len(nodes)
        args.rank = int(os.environ['SLURM_PROCID'])

    except KeyError as e:
        # We are NOT using SLURM
        args.world_size = 1
        args.rank = 0
        nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method('forkserver')

        print(args.rank)
        port = np.random.randint(15000, 15025)
        args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(args.dist_url)
        args.dist_backend = 'nccl'
        args.gpu = None

    # Node here refers to number of computers so this gives you the number of
    # GPUs per computer
    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers  # Dataloader workers
    args.ngpus_per_node = ngpus_per_node

    config = {
        'lr': tune.loguniform(1e-5, 1e-3),
        'chamfer': tune.loguniform(0.01, 1),
        'weight_decay': tune.loguniform(0.01, 0.9),
        'div_factor': tune.loguniform(10,100),
        'final_div_factor': tune.loguniform(2, 1000),
        'n_bins': tune.choice([256, 512, 1024]),
    }

    num_samples = 60

    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=20,
        grace_period=8,
        reduction_factor=2
    )

    reporter = CLIReporter(
        parameter_columns = ['lr', 'chamfer', 'weight_decay',
                             'div_factor', 'final_div_factor', 'n_bins'],
        metric_columns = ['loss', 'accuracy', 'training_iteration']
    )

    if args.cpu:
        main_worker(None, ngpus_per_node, args)

    else:
        if args.distributed:
            # total number of GPUs across all nodes/computers
            args.world_size = ngpus_per_node * args.world_size
            mp.spawn(
                main_worker,
                nprocs=ngpus_per_node,
                args=(
                    ngpus_per_node,
                    args))
        else:
            if torch.cuda.device_count() > 1:
                args.gpu = 1 if torch.cuda.mem_get_info(0)[0] < 9000000000 else 0
            else:
                args.gpu = 0
            
            ngpus_per_node = 1      #forcing single GPU runs to enable multiple concurrent raytune runs

            #if ngpus_per_node == 1:
                # Since only one GPU, set device index to 0 (first and only GPU)
            #    args.gpu = 0
            
            result = tune.run(
                partial(main_worker, gpu=args.gpu, ngpus_per_node=ngpus_per_node, args=args),
                resources_per_trial={'cpu':8, 'gpu':1},
                config=config,
                num_samples=num_samples,
                scheduler=scheduler,
                progress_reporter=reporter,
                local_dir=args.log_dir,
                log_to_file=True,
                checkpoint_at_end=False
            )

            best_trial = result.get_best_trial("accuracy", "max", "last")
            print(f"Best Trial Config {best_trial.config}")
            print(f"Best Trial Best {best_trial.last_result['loss']}")
            print(f"Best Trial Final Validation Accuracy{best_trial.last_result['accuracy']}")
            #main(rank=0, world_size=1, arguments=arguments, config=config)


