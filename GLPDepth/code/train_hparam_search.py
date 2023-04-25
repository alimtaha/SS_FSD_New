import os
from uuid import UUID
import cv2
from imageio import save
import numpy as np
from datetime import datetime
from collections import OrderedDict
import sys


import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tensorboardX import SummaryWriter

from models.model import GLPDepth
import utils.metrics as metrics
from utils.criterion import SiLogLoss
import utils.logging as logging
from utils.lr_decay import lr_decay
from PIL import Image

from dataset.base_dataset import get_dataset
from configs.train_options import TrainOptions
from torch.utils.data.distributed import DistributedSampler

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog', 'iter']


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


def setup (rank, world_size, dataset, batch_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, 
                                shuffle=False, drop_last=False)

    return sampler

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, arguments):

    args = arguments

    #distributed set up
    distributed = True if world_size > 1 else False 
    
    model = GLPDepth(max_depth=args.max_depth, is_train=True)

    if args.ckpt_dir != 'None':
        model_weight = torch.load(args.ckpt_dir)
        if 'module' in next(iter(model_weight.items()))[0]:
            model_weight = OrderedDict((k[7:], v)
                                       for k, v in model_weight.items())
        model.load_state_dict(model_weight)

    sampler=None

    # Dataset setting
    dataset_kwargs = {
        'dataset_name': args.dataset,
        'data_path': args.data_path,
        'train_txt': args.train_txt,
        'test_txt': args.test_txt,
        'logarithmic': args.logarithmic,
        'disparity': args.disparity}
    if args.dataset == 'nyudepthv2':
        dataset_kwargs['crop_size'] = (448, 576)
    elif args.dataset == 'kitti':
        dataset_kwargs['crop_size'] = (352, 704)
    else:
        dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

    print(dataset_kwargs['crop_size'])
    train_dataset = get_dataset(**dataset_kwargs)
    val_dataset = get_dataset(**dataset_kwargs, is_train=False)

    if distributed:
        sampler = setup(rank=rank, world_size=world_size, dataset=train_dataset, batch_size=args.batch_size)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.to(rank)
        model = DDP(model, device_ids=[rank],output_device=rank,
                    find_unused_parameters=False)
        device = rank

    elif args.gpu_or_cpu == 'gpu':  # CPU-GPU agnostic settings
        
        if torch.cuda.device_count() > 1:
            device = 1 if torch.cuda.mem_get_info(0)[0] < 9000000000 else 0
        else:
            device = 0

        device = torch.device(f'cuda:{device}')
        
        #if torch.cuda.device_count() > 1:
        #    model = torch.nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
        model.to(device)

    else:
        device = torch.device('cpu')
        model.to(device)


    if rank == 0:
        # Logging
        print(args)
        date_time = (datetime.now().strftime('%m%d_%H%M'))
        experiment_name = 'GLP' + str(args.epochs) + 'Epochs_' + str(args.lr) + 'LR_' + str(args.crop_h) + 'x' + str(
            args.crop_w) + 'crop_' + str(args.max_depth) + 'max_depth' + str(args.logarithmic) + '_Log'
        log_dir = os.path.join(args.log_dir, date_time, experiment_name)
        tb_dir = os.path.join(log_dir, experiment_name)
        print('\nLogging Dir\n', tb_dir)
        logging.check_and_make_dirs(log_dir)
        logging.check_and_make_dirs(tb_dir)
        writer = SummaryWriter(logdir=tb_dir)
        log_txt = os.path.join(log_dir, 'logs.txt')
        logging.log_args_to_txt(log_txt, args)

        global result_dir
        result_dir = os.path.join(log_dir, 'results')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    else:
        writer = None
    
    if distributed:
        dist.barrier()
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if distributed else args.workers,
        pin_memory=False if distributed else True,
        sampler=sampler,
        drop_last=True)
    if rank == 0:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    # Training settings
    criterion_d = SiLogLoss(max_depth=args.max_depth)
    optimizer = optim.Adam(model.parameters(), args.lr)

    global global_step
    global_step = 0
    
    if rank == 0:
        global best_loss
        best_loss = 100000

    # Perform experiment
    for epoch in range(1, args.epochs + 1):
        print('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        
        if distributed:
            train_loader.sampler.set_epoch(epoch)
        loss_train = train(
            train_loader,
            model,
            criterion_d,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            writer=writer,
            rank=rank,
            world_size=world_size, 
            distributed=distributed)
        #writer.add_scalar('Training loss', loss_train, epoch)

        if epoch % args.val_freq == 0 and rank == 0:
            results_dict, loss_val = validate(val_loader, model, criterion_d,
                                              device=device, epoch=epoch, args=args,
                                              log_dir=log_dir, distributed=distributed)
            writer.add_scalar('Val loss', loss_val, epoch)

            result_lines = logging.display_result(results_dict)
            if args.kitti_crop:
                print("\nCrop Method: ", args.kitti_crop)
            print(result_lines)

            with open(log_txt, 'a') as txtfile:
                txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
                txtfile.write(result_lines)

            for mask_name, res_dict in results_dict.items():
                if res_dict['iter'] > 0:
                    for each_metric, each_results in res_dict.items():
                        writer.add_scalar(
                            f'{mask_name}/{each_metric}', each_results, epoch)
        if distributed:
            dist.barrier()
    
    cleanup()

def train(
        train_loader,
        model,
        criterion_d,
        optimizer,
        device,
        epoch,
        args,
        writer,
        rank,
        world_size,
        distributed):
    
    global global_step
    model.train()
    depth_loss = logging.AverageMeter()
    half_epoch = args.epochs // 2

    for batch_idx, batch in enumerate(train_loader):

        global_step += (1 * world_size)

        if args.debug and global_step > 10:
            loss_d = 0
            break

        for param_group in optimizer.param_groups:
            """
            if global_step < 2019 * half_epoch:
                current_lr = (1e-4 - 3e-5) * (global_step /
                                              2019/half_epoch) ** 0.9 + 3e-5
            else:
                current_lr = (3e-5 - 1e-4) * (global_step /
                                              2019/half_epoch - 1) ** 0.9 + 1e-4
            """
            current_lr = lr_decay(global_step, args.lr, 1e-7, 0.999)
            param_group['lr'] = current_lr

            #OPTIMAL TRAINING!
            #2e-4 seems to be optimal learning rate on 10 epochs with 0.9999 lr decay poly
        
        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)

        preds = model(input_RGB)

        optimizer.zero_grad()
        loss_d = criterion_d(preds['pred_d'].squeeze(), depth_gt)
        depth_loss.update(loss_d.item(), input_RGB.size(0))
        loss_d.backward()

        if global_step % 20 == 0 and rank == 0:
            writer.add_scalar('Training loss', loss_d, global_step)

        logging.progress_bar(batch_idx, len(train_loader), args.epochs, epoch,
                             ('Depth Loss: %.4f (%.4f) - Current lr: (%.4E) Initial lr (%.4E)' %
                              (depth_loss.val, depth_loss.avg, current_lr, args.lr)))
        optimizer.step()
    

    return loss_d


def validate(val_loader, model, criterion_d, device, epoch, args, log_dir, distributed):
    depth_loss = logging.AverageMeter()
    global best_loss
    model.eval()

    masks = ['mask_20', 'mask_80', 'mask_20_80', 'mask_80_256', 'mask_all']

    result_metrics = {}

    for mask in masks:
        result_metrics[mask] = {}
        for metric in metric_name:
            result_metrics[mask][metric] = 0.0

    '''result_metrics_80 = {}
    for metric in metric_name:
        result_metrics_80[metric] = 0.0

    result_metrics_20_80 = {}
    for metric in metric_name:
        result_metrics_20_80[metric] = 0.0

    result_metrics_80_256 = {}
    for metric in metric_name:
        result_metrics_80_256[metric] = 0.0

    result_metrics_all = {}
    for metric in metric_name:
        result_metrics_all[metric] = 0.0'''

    #results = [result_metrics_20, result_metrics_80, result_metrics_20_80, result_metrics_80_256, result_metrics_all]

    for batch_idx, batch in enumerate(val_loader):
        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        filename = batch['filename'][0]

        with torch.no_grad():
            preds = model(input_RGB)

        pred_d = preds['pred_d'].squeeze()
        depth_gt = depth_gt.squeeze()

        if args.logarithmic:
            pred_d = (torch.exp(4 * torch.log(torch.tensor(1.01))
                                * (pred_d + 50)) - 1.01**200)
            # depth_gt from dataloader doesn't need to be converted from non
            # log space since eval dataset/dataloader don't perform log space
            # conversion

        loss_d = criterion_d(preds['pred_d'].squeeze(), depth_gt)

        depth_loss.update(loss_d.item(), input_RGB.size(0))

        pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)
        save_path = result_dir #os.path.join(result_dir, 'latest_pred_maps')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        #test_path = '/home/extraspace/Logs/Testing_save/'
        save_path = os.path.join(save_path, filename)

        if save_path.split('.')[-1] == 'jpg':
            save_path = save_path.replace('jpg', 'png')

        if args.save_result:
            if args.dataset == 'kitti' or args.dataset == 'cityscapes':
                pred_d_numpy = pred_d.cpu().numpy() * 256.0
                cv2.imwrite(
                    save_path, pred_d_numpy.astype(
                        np.uint16), [
                        cv2.IMWRITE_PNG_COMPRESSION, 0])

            else:
                pred_d_numpy = pred_d.cpu().numpy() * 1000.0
                cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])

        loss_d = depth_loss.avg
        logging.progress_bar(batch_idx, len(val_loader), args.epochs, epoch)

        for mask in result_metrics.keys():
            if mask in computed_result.keys():
                for key in result_metrics[mask].keys():
                    result_metrics[mask][key] += computed_result[mask][key]

    for mask in result_metrics.keys():
        if result_metrics[mask]['iter'] > 0:
            for key in result_metrics[mask].keys():
                if key != 'iter':
                    result_metrics[mask][key] /= result_metrics[mask]['iter']

    if args.save_model:
        if distributed:
            torch.save(model.module.state_dict(), os.path.join(
                       log_dir, 'latest_model.ckpt'))
            if loss_d < best_loss:
                torch.save(model.module.state_dict(), os.path.join(
                       log_dir, 'best_loss_model.ckpt'))
                best_loss = loss_d
        else:
            torch.save(model.state_dict(), os.path.join(
                       log_dir, 'latest_model.ckpt'))
            if loss_d < best_loss or best_loss:
                torch.save(model.state_dict(), os.path.join(
                       log_dir, 'best_loss_model.ckpt'))
                best_loss = loss_d

    return result_metrics, loss_d

if __name__ == '__main__':
    
    opt = TrainOptions()
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        arguments = opt.initialize()
        arguments.convert_arg_line_to_args = convert_arg_line_to_args
        arguments = arguments.parse_args([arg_filename_with_prefix])
    else:
        arguments = opt.initialize().parse_args()
    
    if torch.cuda.device_count() > 1 and arguments.dist is True:
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size, arguments), nprocs=world_size)
    else:
        main(rank=0, world_size=1, arguments=arguments)