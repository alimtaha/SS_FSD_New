from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

import os
import sys
import time
from datetime import datetime
from telnetlib import IP
import argparse
import uuid
import cv2
import numpy as np
from tqdm import tqdm


sys.path.append('../')
sys.path.append('../../')
from torch.utils.tensorboard import SummaryWriter
from utils import post_process_depth, flip_lr, silog_loss, compute_errors, compute_errors_all, eval_metrics, \
    block_print, enable_print, normalize_result, inv_normalize, convert_arg_line_to_args
from networks.NewCRFDepth import NewCRFDepth

new_config = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/SS_FSD_New/NeWCRFs/hyperparams/configurable.txt'     #AT Hyperparam - new config and parsing using environ variables

with open(sys.argv[1], 'r') as c: #AT Hyperparam
    cfg_file = c.readlines()

width_value = os.environ['H_WIDTH']
#height_value = 704 if int(width_value) == 352 else 352
height_value = os.environ['H_HEIGHT']
max_depth_value = str(os.environ["H_MAXDEPTH"])
lr_value = str(os.environ["H_LR"])
dataset_value = str(os.environ["H_DATASET"])

with open(new_config, 'w') as w:
    for idx, _ in enumerate(cfg_file):
        cfg_file[idx] = cfg_file[idx].replace("H_MAXDEPTH", max_depth_value)
        cfg_file[idx] = cfg_file[idx].replace("H_LR", lr_value)
        cfg_file[idx] = cfg_file[idx].replace("H_WIDTH", str(width_value))
        cfg_file[idx] = cfg_file[idx].replace("H_HEIGHT", str(height_value))
        cfg_file[idx] = cfg_file[idx].replace("H_DATASET", dataset_value)
    w.writelines(cfg_file)

parser = argparse.ArgumentParser(
    description='NeWCRFs PyTorch implementation.',
    fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode', type=str, help='train or test', default='train')
parser.add_argument(
    '--model_name',
    type=str,
    help='model name',
    default='newcrfs')
parser.add_argument(
    '--encoder',
    type=str,
    help='type of encoder, base07, large07',
    default='large07')
parser.add_argument(
    '--pretrain',
    type=str,
    help='path of pretrained encoder',
    default=None)

# Dataset
parser.add_argument(
    '--dataset',
    type=str,
    help='dataset to train on, kitti or nyu',
    default='nyu')
parser.add_argument(
    '--data_path',
    type=str,
    help='path to the data',
    required=True)
parser.add_argument(
    '--gt_path',
    type=str,
    help='path to the groundtruth data',
    required=True)
parser.add_argument(
    '--camera_path',
    type=str,
    help='path to the groundtruth camera parameters',
    required=False)
parser.add_argument(
    '--disparity_path',
    type=str,
    help='path to the groundtruth camera parameters',
    required=False)    
parser.add_argument(
    '--disparity',
    action='store_true',
    help='use disparity vs depth maps',
    required=False) 
parser.add_argument(
    '--filenames_file',
    type=str,
    help='path to the filenames text file',
    required=True)
parser.add_argument(
    '--sampled_file',              
    type=str,   
    help='path to the sampled filenames text file', 
    required=True)
parser.add_argument(
    '--input_height',
    type=int,
    help='input height',
    default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument(
    '--max_depth',
    type=float,
    help='maximum depth in estimation',
    default=10)
parser.add_argument(
    '--min_depth_train',
    type=float,
    help='minimum depth in training mask',
    default=0.001)

# Log and save
parser.add_argument(
    '--log_directory',
    type=str,
    help='directory to save checkpoints and summaries',
    default='')
parser.add_argument(
    '--checkpoint_path',
    type=str,
    help='path to a checkpoint to load',
    default='')
parser.add_argument(
    '--log_freq',
    type=int,
    help='Logging frequency in global steps',
    default=100)
parser.add_argument(
    '--save_freq',
    type=int,
    help='Checkpoint saving frequency in global steps',
    default=5000)
parser.add_argument(
    '--model_save',
    action='store_true',
    help='store best checkpoints',
    required=False)

# Training
parser.add_argument(
    '--weight_decay',
    type=float,
    help='weight decay factor for optimization',
    default=1e-2)
parser.add_argument(
    '--retrain',
    help='if used with checkpoint_path, will restart training from step zero',
    action='store_true')
parser.add_argument(
    '--adam_eps',
    type=float,
    help='epsilon in Adam optimizer',
    default=1e-6)
parser.add_argument('--batch_size', type=int, help='batch size', default=4)
parser.add_argument(
    '--num_epochs',
    type=int,
    help='number of epochs',
    default=50)
parser.add_argument(
    '--learning_rate',
    type=float,
    help='initial learning rate',
    default=1e-4)
parser.add_argument(
    '--end_learning_rate',
    type=float,
    help='end learning rate',
    default=-1)
parser.add_argument(
    '--variance_focus',
    type=float,
    help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error',
    default=0.85)
parser.add_argument(
    '--sampled',                            
    help='if set, uses sampled train_extra dataset', 
    action='store_true')

# Preprocessing
parser.add_argument(
    '--do_random_rotate',
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
    '--cpu',
    help='if set, will use CPU for training',
    action='store_true')

# Multi-gpu training
parser.add_argument(
    '--num_threads',
    type=int,
    help='number of threads to use for data loading',
    default=1)
parser.add_argument(
    '--world_size',
    type=int,
    help='number of nodes for distributed training',
    default=1)
parser.add_argument(
    '--rank',
    type=int,
    help='node rank for distributed training',
    default=0)
parser.add_argument(
    '--dist_url',
    type=str,
    help='url used to set up distributed training',
    default='tcp://127.0.0.1:1234')
parser.add_argument(
    '--dist_backend',
    type=str,
    help='distributed backend',
    default='nccl')
parser.add_argument('--gpu', type=int, help='GPU id to use.', default=None)
parser.add_argument(
    '--multiprocessing_distributed',
    help='Use multi-processing distributed training to launch '
    'N processes per node, which has N GPUs. This is the '
    'fastest way to use PyTorch for either single node or '
    'multi node data parallel training',
    action='store_true',
)
# Online eval
parser.add_argument(
    '--do_online_eval',
    help='if set, perform online eval in every eval_freq steps',
    action='store_true')
parser.add_argument(
    '--save_eval',
    help='if set, perform online eval in every eval_freq steps',
    action='store_true')
parser.add_argument(
    '--data_path_eval',
    type=str,
    help='path to the data for online evaluation',
    required=False)
parser.add_argument(
    '--gt_path_eval',
    type=str,
    help='path to the groundtruth data for online evaluation',
    required=False)
parser.add_argument(
    '--filenames_file_eval',
    type=str,
    help='path to the filenames text file for online evaluation',
    required=False)
parser.add_argument(
    '--min_depth_eval',
    type=float,
    help='minimum depth for evaluation',
    default=1e-3)
parser.add_argument(
    '--max_depth_eval',
    type=float,
    help='maximum depth for evaluation',
    default=80)
parser.add_argument(
    '--eigen_crop',
    help='if set, crops according to Eigen NIPS14',
    action='store_true')
parser.add_argument(
    '--garg_crop',
    help='if set, crops according to Garg  ECCV16',
    action='store_true')
parser.add_argument(
    '--eval_freq',
    type=int,
    help='Online evaluation frequency in global steps',
    default=500)
parser.add_argument(
    '--eval_summary_directory',
    type=str,
    help='output directory for eval summary,'
    'if empty outputs to checkpoint folder',
    default='')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + new_config
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu' or args.dataset == 'cityscapes':
    from dataloaders.dataloader import NewDataLoader
elif args.dataset == 'kittipred':
    from dataloaders.dataloader_kittipred import NewDataLoader

if args.sampled:
    args.filenames_file = args.sampled_file

args.model_name = (

    str(datetime.now().strftime('%m%d_%H%M')) 
    +  'maxdepth:' + max_depth_value 
    + '_width:' + str(width_value) 
    + '_height:' + str(height_value)
    + '_lr:' + lr_value 
    + '_' + dataset_value
)

#AT comment - used for directory naming for logging/saving

print(args.gt_path.split('/')[-2])

def online_eval(model, dataloader_eval, gpu, ngpus, post_process=False):
    eval_measures = torch.zeros(10).cuda(device=gpu)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        
        #TODO: Add masks for evaluation for different depth intervals
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda(gpu, non_blocking=True))
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                # print('Invalid depth. continue.')
                continue

            pred_depth = model(image)
            if post_process:
                image_flipped = flip_lr(image)
                pred_depth_flipped = model(image_flipped)
                pred_depth = post_process_depth(pred_depth, pred_depth_flipped)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        measures = compute_errors_all(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda(device=gpu)
        eval_measures[9] += 1

    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', post_process)
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
        for i in range(8):
            print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.4f}'.format(eval_measures_cpu[8]))
        return eval_measures_cpu

    return None

def main_worker(rank, ngpus_per_node, args, world_size=1, save_path=None):
    args.gpu = None

    if args.gpu is not None:
        print("== Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        '''
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank)
        '''
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, 
                                shuffle=False, drop_last=False)

    # NeWCRFs model
    model = NewCRFDepth(
        version=args.encoder,
        inv_depth=False,
        max_depth=args.max_depth,
        pretrained=args.pretrain)
    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    if args.distributed:
        if args.gpu is not None: #unused
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.to(rank)
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[rank],output_device=rank,
                    find_unused_parameters=False)
            device = rank
    elif args.cpu == False:
        
        if torch.cuda.device_count() > 1:
            device = 1 if torch.cuda.mem_get_info(0)[0] < 9000000000 else 0
        else:
            device = 0
        model = torch.nn.DataParallel(model)
        device = torch.device(f'cuda:{device}')
        
    else:
        device='cpu'

    model.to(device)

    if args.distributed:
        print("== Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("== Model Initialized")

    global_step = 0
    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)

    # Training parameters
    optimizer = torch.optim.Adam([{'params': model.module.parameters()}],
                                 lr=args.learning_rate)

    model_just_loaded = False
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            model.load_state_dict(checkpoint['model'])

            if not args.retrain:
                try:
                    global_step = checkpoint['global_step']
                    best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu(
                    )
                    best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu(
                    )
                    best_eval_steps = checkpoint['best_eval_steps']
                    print("== Loaded checkpoint '{}' (global_step {})".format(
                        args.checkpoint_path, checkpoint['global_step']))
                    del checkpoint
                except KeyError:
                    print("Could not load values for online evaluation")
        else:
            print(
                "== No checkpoint found at '{}'".format(
                args.checkpoint_path))
        model_just_loaded = True
        

    cudnn.benchmark = True

    dataloader = NewDataLoader(args, 'train', world_size=world_size, rank=rank, shuffle=False, drop_last=False)
    
    if rank == 0:
        dataloader_eval = NewDataLoader(args, 'online_eval', world_size=world_size, rank=rank, shuffle=True, drop_last=False)

    # ===== Evaluation before training ======
    # model.eval()
    # with torch.no_grad():
    #     eval_measures = online_eval(model, dataloader_eval, gpu, ngpus_per_node, post_process=True)


    # Logging
    if rank == 0:
        
        writer = SummaryWriter(save_path + '/summaries', flush_secs=30)

        #if args.do_online_eval:
            #if args.eval_summary_directory != '':
            #    eval_summary_path = os.path.join(
            #        args.eval_summary_directory, args.model_name)
            #else:
            #    eval_summary_path = os.path.join(
            #        args.log_directory, args.model_name, 'eval')
            #eval_summary_writer = SummaryWriter(
            #    eval_summary_path, flush_secs=30)
    
    hparams_dict = {
        'lr': args.learning_rate,
        'adam_eps': args.adam_eps,
        'height': args.input_height,
        'width': args.input_width,
        'max_depth': args.max_depth,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'dset_type': args.gt_path.split('/')[-2]
    }

    if args.distributed:
        torch.distributed.parallel.barrier()

    silog_criterion = silog_loss(variance_focus=args.variance_focus)

    start_time = time.time()
    duration = 0

    num_log_images = args.batch_size
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != - \
        1 else 0.1 * args.learning_rate

    var_sum = [var.sum().item()
               for var in model.parameters() if var.requires_grad]
    var_cnt = len(var_sum)
    var_sum = np.sum(var_sum)

    print(
        "== Initial variables' sum: {:.3f}, avg: {:.3f}".format(
            var_sum,
            var_sum /
            var_cnt))

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    epoch = global_step // steps_per_epoch

    while epoch < args.num_epochs:
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        for step, sample_batched in tqdm(
            enumerate(
                dataloader.data), desc=f"Epoch: {epoch + 1}/{args.num_epochs}. Loop: Train", total=len(
                dataloader.data)):
            optimizer.zero_grad()
            before_op_time = time.time()

            if args.cpu == False:
                image = torch.autograd.Variable(
                    sample_batched['image'].cuda(
                        args.gpu, non_blocking=True))
                depth_gt = torch.autograd.Variable(
                    sample_batched['depth'].cuda(
                        args.gpu, non_blocking=True))
            else:
                image = torch.autograd.Variable(
                    sample_batched['image'].cpu())
                depth_gt = torch.autograd.Variable(
                    sample_batched['depth'].cpu())

            if args.dataset == 'nyu':
                mask = depth_gt > 0.1      #changed from 0.01
            else:
                mask = depth_gt > 1.0     #changed from 0.001

            mask_max = (depth_gt <= args.max_depth)
            mask = torch.logical_and(mask, mask_max)

            depth_est = model(image*mask.float())       #test computational efficiency if masking done before feeding into network

            '''
            b, c, h, w = depth_gt.shape
            
            for i in range(b):
                h_reduced = h // 4
                w_reduced = w // 4

                masks_downsample = np.ones(h_reduced * w_reduced)
                indices = np.arange(0, h_reduced * w_reduced)
                indices = np.random.choice(indices, size= h_reduced * w_reduced // 2, replace=False)

                masks_downsample[indices] = 0
                masks_downsample = masks_downsample.reshape(h_reduced, w_reduced).astype(int)
                masks_downsample = cv2.resize(masks_downsample, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
                masks_downsample = masks_downsample[np.newaxis, np.newaxis, ...]

                if i == 0:
                    masks_downsample_concat = masks_downsample
                else:
                    masks_downsample_concat = np.concatenate((masks_downsample_concat, masks_downsample), axis=0)

            masks_downsample_concat = torch.from_numpy(masks_downsample_concat).cuda()

            mask = torch.logical_and(mask, masks_downsample_concat)
            '''


            loss = silog_criterion.forward(
                depth_est, depth_gt, mask.to(torch.bool))
            loss.backward()
            for param_group in optimizer.param_groups:
                current_lr = (args.learning_rate - end_learning_rate) * \
                    (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                param_group['lr'] = current_lr

            optimizer.step()

            if rank == 0 and step % 20 == 0:
                print('[epoch][s/s_per_e/gs]: [{}][{}/{}/{}], lr: {:.12f}, loss: {:.12f}'.format(
                    epoch, step, steps_per_epoch, global_step, current_lr, loss))
                if np.isnan(loss.cpu().item()):
                    print('NaN in loss occurred. Aborting training.')
                    return -1

            duration += time.time() - before_op_time
            if global_step and global_step % args.log_freq == 0 and not model_just_loaded and rank == 0:
                var_sum = [var.sum().item()
                           for var in model.parameters() if var.requires_grad]
                var_cnt = len(var_sum)
                var_sum = np.sum(var_sum)
                examples_per_sec = args.batch_size / duration * args.log_freq
                duration = 0
                time_sofar = (time.time() - start_time) / 3600
                training_time_left = (
                    num_total_steps / global_step - 1.0) * time_sofar
                
                if rank == 0:
                    print("{}".format(args.model_name))
                
                print_string = 'GPU: {} | examples/s: {:4.2f} | loss: {:.5f} | var sum: {:.3f} avg: {:.3f} | time elapsed: {:.2f}h | time left: {:.2f}h | current lr: {:.3E} | initial lr: {:.3E}'
                
                print(
                    print_string.format(
                        args.gpu,
                        examples_per_sec,
                        loss,
                        var_sum.item(),
                        var_sum.item() /
                        var_cnt,
                        time_sofar,
                        training_time_left,
                        current_lr,
                        args.learning_rate))

                if rank == 0:
                    writer.add_scalar('silog_loss', loss, global_step)
                    writer.add_scalar('learning_rate', current_lr, global_step)
                    writer.add_scalar(
                        'var average', var_sum.item() / var_cnt, global_step)
                    depth_gt = torch.where(
                        depth_gt < 1e-3, depth_gt * 0 + 1e3, depth_gt)
                    for i in range(num_log_images):
                        writer.add_image(
                            'depth_gt/image/{}'.format(i), normalize_result(1 / depth_gt[i, :, :, :].data), global_step)
                        writer.add_image('depth_est/image/{}'.format(i), normalize_result(
                            1 / depth_est[i, :, :, :].data), global_step)
                        writer.add_image(
                            'image/image/{}'.format(i), inv_normalize(image[i, :, :, :]).data, global_step)
                    writer.flush()
                    #TODO change the normalising function to log images with 256 depth

            if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
                time.sleep(0.1)
                model.eval()
                with torch.no_grad():
                    eval_measures = online_eval(model, dataloader_eval, rank, ngpus_per_node, post_process=True)
                if eval_measures is not None:
                    for i in range(9):
                        writer.add_scalar('val/' + eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                        measure = eval_measures[i]
                        is_best = False
                        if i < 6 and measure < best_eval_measures_lower_better[i]:
                            old_best = best_eval_measures_lower_better[i].item()
                            best_eval_measures_lower_better[i] = measure.item()
                            is_best = True
                        elif i >= 6 and measure > best_eval_measures_higher_better[i-6]:
                            old_best = best_eval_measures_higher_better[i-6].item()
                            best_eval_measures_higher_better[i-6] = measure.item()
                            is_best = True
                        if is_best:
                            old_best_step = best_eval_steps[i]
                            old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[i], old_best)
                            #model_path = args.log_directory + '/' + args.model_name + old_best_name
                            #if os.path.exists(model_path):
                            #    command = 'rm {}'.format(model_path)
                            #    os.system(command)
                            best_eval_steps[i] = global_step
                            #model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, eval_metrics[i], measure)
                            #print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                            #checkpoint = {'model': model.state_dict()}
                            #torch.save(checkpoint, args.log_directory + '/' + 'hyperparameters' + '/' + args.model_name + model_save_name)
                    writer.flush()
                model.train()
                block_print()
                enable_print()

            model_just_loaded = False
            global_step += 1

        epoch += 1

    hparam_metric_dict = { #AT Hyperparam
        'd1': best_eval_measures_higher_better[0],
        'd2': best_eval_measures_higher_better[1],
        'd3': best_eval_measures_higher_better[2],
        'rms': best_eval_measures_lower_better[3],
        'log_rms':best_eval_measures_lower_better[5],
        'abs_rel':best_eval_measures_lower_better[1],
        'sq_rel': best_eval_measures_lower_better[4],
        'silog' :best_eval_measures_lower_better[0],
        'log10': best_eval_measures_lower_better[2]

    }
    
    #Best metrics logged against hyperparams
    writer.add_hparams(hparams_dict, hparam_metric_dict)

    #if not args.multiprocessing_distributed or (
    #        args.multiprocessing_distributed and args.rank %
    #        ngpus_per_node == 0):
        #writer.close()
        #if args.do_online_eval:
            #eval_summary_writer.close()


def main():
    if args.mode != 'train':
        print('train.py is only for training.')
        return -1

    # date_time = (datetime.now().strftime('%m%d_%H%M'))
    # subdir = args.log_directory + '/' + date_time + '/' +  args.encoder + '/' + experiment_name  
    # args.log_directory = os.path.join(args.log_directory, subdir)
    # tb_dir = os.path.join(args.log_directory, '/summaries')
    

    command = 'mkdir ' + os.path.join(args.log_directory, 'hyperparameters', args.model_name)
    os.system(command)

    args_out_path = os.path.join(args.log_directory, 'hyperparameters', args.model_name)
    #command = 'cp ' + sys.argv[1] + ' ' + args_out_path #AT Hyperparam
    command = 'cp ' + new_config + ' ' + args_out_path #AT Hyperparam
    os.system(command)

    save_files = True
    if save_files:
        aux_out_path = os.path.join(args.log_directory, 'hyperparameters', args.model_name)
        networks_savepath = os.path.join(aux_out_path, 'networks')
        dataloaders_savepath = os.path.join(aux_out_path, 'dataloaders')
        config_savepath = os.path.join(aux_out_path, 'configs')
        command = 'cp ./newcrfs/hyperparam_train.py ' + aux_out_path
        os.system(command)
        command = 'mkdir -p ' + networks_savepath + ' && cp /home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/SS_FSD_New/NeWCRFs/newcrfs/networks/*.py ' + networks_savepath
        os.system(command)
        command = 'mkdir -p ' + dataloaders_savepath + ' && /home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/SS_FSD_New/NeWCRFs/newcrfs/dataloaders/*.py ' + dataloaders_savepath
        os.system(command)

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 and args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        main_worker(rank=0, ngpus_per_node=ngpus_per_node, args=args, save_path=aux_out_path)
        
    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print(
            "This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics." .format(
                args.eval_freq))

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node #* args.world_size
        mp.spawn(
            main_worker,
            ngpus_per_node=ngpus_per_node,
            args=(args,),
            save_path=aux_out_path
            )
    else:
        main_worker(rank=0, ngpus_per_node=ngpus_per_node, args=args, save_path=aux_out_path)


if __name__ == '__main__':
    main()
