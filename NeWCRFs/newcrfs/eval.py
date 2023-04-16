import torch
import torch.backends.cudnn as cudnn

from datetime import datetime
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import post_process_depth, flip_lr, compute_errors, compute_errors_all
from networks.NewCRFDepth import NewCRFDepth


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(
    description='NeWCRFs PyTorch implementation.',
    fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

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
    '--checkpoint_path',
    type=str,
    help='path to a checkpoint to load',
    default='')

# Dataset
parser.add_argument(
    '--dataset',
    type=str,
    help='dataset to train on, kitti or nyu',
    default='nyu')
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
parser.add_argument('--semantic_labels_dataset',
    type=str,   
    help='semantic labels to be used for evaluation on road masks only', default='')


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

# Eval
parser.add_argument(
    '--data_path_eval',
    type=str,
    help='path to the data for evaluation',
    required=False)
parser.add_argument(
    '--gt_path_eval',
    type=str,
    help='path to the groundtruth data for evaluation',
    required=False)
parser.add_argument(
    '--filenames_file_eval',
    type=str,
    help='path to the filenames text file for evaluation',
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
parser.add_argument('--road_mask',
    help='if set, only uses road masks for evaluation', 
    action='store_true')



if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu' or args.dataset == 'cityscapes':
    from dataloaders.dataloader import NewDataLoader
elif args.dataset == 'kittipred':
    from dataloaders.dataloader_kittipred import NewDataLoader


def eval(model, dataloader_eval, post_process=False):
    #eval_measures = torch.zeros(10).cuda(device=gpu)
    masks = ['mask_20', 'mask_80', 'mask_20_80', 'mask_80_256', 'mask_all']
    metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rms', 'log_rms',
               'log10', 'silog', 'iter']

    result_metrics = {}

    for mask in masks:
        result_metrics[mask] = {}
        for metric in metric_name:
            result_metrics[mask][metric] = 0.0
            
    # eval_measures = torch.zeros(10).cuda()
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = eval_sample_batched['image'].cuda()
            gt_depth = eval_sample_batched['depth']
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if args.road_mask:
                road_mask = eval_sample_batched['road_mask']
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
            road_mask = road_mask.cpu().numpy().squeeze()

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352,
                                 left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(
            gt_depth > args.min_depth_eval,
            gt_depth < args.max_depth_eval)
        
        if args.road_mask:
            valid_mask = np.logical_and(valid_mask, road_mask.astype(np.bool))

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 *
                              gt_height):int(0.99189189 *
                                             gt_height), int(0.03594771 *
                                                             gt_width):int(0.96405229 *
                                                                           gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 *
                                  gt_height):int(0.91351351 *
                                                 gt_height), int(0.0359477 *
                                                                 gt_width):int(0.96405229 *
                                                                               gt_width)] = 1
                elif args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        for mask in measures.keys():
            if mask in measures.keys():
                for key in result_metrics[mask].keys():
                        result_metrics[mask][key] += measures[mask][key]

    for mask in result_metrics.keys():
            if result_metrics[mask]['iter'] > 0:
                for key in result_metrics[mask].keys():
                    if key != 'iter':
                        result_metrics[mask][key] /= result_metrics[mask]['iter']

        # measures = compute_errors_all(gt_depth[valid_mask], pred_depth[valid_mask])

        # eval_measures[:9] += torch.tensor(measures).cuda()
        # eval_measures[9] += 1

    # eval_measures_cpu = eval_measures.cpu()
    # cnt = eval_measures_cpu[9].item()
    # eval_measures_cpu /= cnt
    # print('Computing errors for {} eval samples'.format(
    #     int(cnt)), ', post_process: ', post_process)
    # print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
    #     'silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3'))
    # for i in range(8):
    #     print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
    # print('{:7.4f}'.format(eval_measures_cpu[8]))
    # return eval_measures_cpu

    if not args.distributed or args.rank == 0:
        for masks, mask_dict in result_metrics.items():
                print(f"{masks} - Validated: \n {mask_dict}\n\n")

    return result_metrics


def main_worker(args):

    # CRF model
    model = NewCRFDepth(
        version=args.encoder,
        inv_depth=False,
        max_depth=args.max_depth,
        pretrained=None)
    model.train()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("== Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape)
                             for p in model.parameters() if p.requires_grad])
    print("== Total number of learning parameters: {}".format(num_params_update))

    model = torch.nn.DataParallel(model)
    model.cuda()

    print("== Model Initialized")

    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("== Loading checkpoint '{}'".format(args.checkpoint_path))
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            print("== Loaded checkpoint '{}'".format(args.checkpoint_path))
            del checkpoint
        else:
            print(
                "== No checkpoint found at '{}'".format(
                    args.checkpoint_path))

    cudnn.benchmark = True

    dataloader_eval = NewDataLoader(args, 'online_eval')

    args.model_name = (
        str(datetime.now().strftime('%m%d_%H%M')) 
        +  'maxdepth:' + str(args.max_depth_eval)
        + 'road_masks:' + str(args.road_mask)
    )
    
    tb_dir = args.log_directory + '/' + args.model_name + '/summaries'
    writer = SummaryWriter(tb_dir, flush_secs=30)

    # ===== Evaluation ======
    model.eval()
    mask_names = [
        'mask_20',
        'mask_80',
        'mask_20_80',
        'mask_80_256',
        'mask_all'
        ]
    with torch.no_grad():
        eval_measures = eval(
                model, dataloader_eval, post_process=True)
    
    flattened_hparam_dict = {}
    if eval_measures is not None:
        for mask, mask_dict in eval_measures.items():
            if eval_measures[mask]['iter'] > 0:
                    for k,v in mask_dict.items():
                        if k != 'iter':
                            key_name = mask + '_' + k
                            flattened_hparam_dict[key_name] = v
                            writer.add_scalar(f'{mask}/{k}', v, 0)

    #Flatten Dict
    hparams_dict = {
        'max_depth': args.max_depth_eval,
        'min_depth': args.min_depth_eval,
        'road_mask': str(args.road_mask),
        'model_name': str(args.checkpoint_path.split('/')[-1])
    }

    # hparam_metric_dict = { 
    #     'd1': eval_measures[6],
    #     'd2': eval_measures[7],
    #     'd3': eval_measures[8],
    #     'rms': eval_measures[3],
    #     'log_rms':eval_measures[5],
    #     'abs_rel':eval_measures[1],
    #     'sq_rel': eval_measures[4],
    #     'silog' :eval_measures[0],
    #     'log10': eval_measures[2]
    # }

    writer.add_hparams(hparams_dict, flattened_hparam_dict)


def main():
    torch.cuda.empty_cache()
    args.distributed = False
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1:
        print("This machine has more than 1 gpu. Please set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    main_worker(args)


if __name__ == '__main__':
    main()
