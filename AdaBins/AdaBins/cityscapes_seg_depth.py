import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import model_io
from dataloader import DepthDataLoader
from models import UnetAdaptiveBins
from utils import RunningAverageDict


def predict_tta(model, image, args):
    # model returns two objects, bin centres and pred, we only want pred so
    # only taking the last output of what model returns
    pred = model(image)[-1]
    # alternatively could have done '_ ,pred = model(image)'
    #     pred = utils.depth_norm(pred)
    #     pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)
    #     pred = np.clip(pred.cpu().numpy(), 10, 1000)/100.
    # clip all predictions to be only between the min and max depth values
    # specified for the dataset
    pred = np.clip(pred.cpu().numpy(), args.min_depth, args.max_depth)

    image = torch.Tensor(np.array(image.cpu().numpy())
                         [..., ::-1].copy()).to(device)

    pred_lr = model(image)[-1]
    #     pred_lr = utils.depth_norm(pred_lr)
    #     pred_lr = nn.functional.interpolate(pred_lr, depth.shape[-2:], mode='bilinear', align_corners=True)
    #     pred_lr = np.clip(pred_lr.cpu().numpy()[...,::-1], 10, 1000)/100.
    # clip all predictions to be only between the min and max depth values
    # specified for the dataset
    pred_lr = np.clip(pred_lr.cpu().numpy()
                      [..., ::-1], args.min_depth, args.max_depth)
    # average of same image but flipped? - mentioned in paper
    final = 0.5 * (pred + pred_lr)
    final = nn.functional.interpolate(torch.Tensor(
        final), image.shape[-2:], mode='bilinear', align_corners=True)
    return torch.Tensor(final)


def eval(model, test_loader, args, gpus=None):

    if gpus is None:
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = gpus[0]

    if args.save_dir is not None:
        os.makedirs(args.save_dir)

    with torch.no_grad():  # reducing gradient computation during inference will allow faster running
        model.eval()  # configuring model to be in eval mode

        sequential = test_loader
        for batch in tqdm(
                sequential):  # dataloader object returns batches of data

            image = batch['image'].to(torch.device('cuda'))
            # only returns the pred image
            final = predict_tta(model, image, args)
            # the image is squeezed into a (1?) dim array for computing all
            # erros
            final = final.squeeze().cpu().numpy()

            # final[final < args.min_depth] = args.min_depth
            # final[final > args.max_depth] = args.max_depth
            # anything thats infinite value is set to the max depth
            final[np.isinf(final)] = args.max_depth
            # anything thats nan value is set to the min depth
            final[np.isnan(final)] = args.min_depth

            if args.save_dir is not None:
                if args.dataset == 'nyu':
                    impath = f"{batch['image_path'][0].replace('/', '__').replace('.jpg', '')}"
                    factor = 1000
                    print('skipping')
                else:
                    impath = batch['image_path'][0]  # depth path derived
                    impath = impath.split('.')[0]  # image path derived
                    factor = 256

                #rgb_path = os.path.join(rgb_dir, f"{impath}.png")
                # tf.ToPILImage()(denormalize(image.squeeze().unsqueeze(0).cpu()).squeeze()).save(rgb_path)
                # path set for saving pred output
                pred_path = os.path.join(args.save_dir, f'{impath}.png')
                # converting them back to a format where they can be compared
                # with the depth maps (so converting the model output from
                # metres to 16bit unsigned int where every bit is 256th of a
                # metre)
                pred = (final * factor).astype('uint16')
                # converts array to a PIL image and saves in pred path
                Image.fromarray(pred).save(pred_path)


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(
        description='Model evaluator',
        fromfile_prefix_chars='@',
        conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument(
        '--n-bins',
        '--n_bins',
        default=256,
        type=int,
        help='number of bins/buckets to divide depth range into')
    parser.add_argument(
        '--gpu',
        default=None,
        type=int,
        help='Which gpu to use')
    parser.add_argument(
        '--save-dir',
        '--save_dir',
        default=None,
        type=str,
        help='Store predictions in folder')
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")

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
    parser.add_argument("--gt_path", default='../dataset/nyu/sync/', type=str,
                        help="path to dataset gt")

    parser.add_argument(
        '--filenames_file',
        default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
        type=str,
        help='path to the filenames text file')

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
        '--do_kb_crop',
        help='if set, crop input images as kitti benchmark images',
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
        '--checkpoint_path',
        '--checkpoint-path',
        type=str,
        required=True,
        help="checkpoint file to use for prediction")

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
        help='if set, crops according to Eigen NIPS14',
        action='store_true')
    parser.add_argument(
        '--garg_crop',
        help='if set, crops according to Garg  ECCV16',
        action='store_true')
    parser.add_argument(
        '--do_kb_crop',
        help='Use kitti benchmark cropping',
        action='store_true')

    if sys.argv.__len__(
    ) == 2:  # length of arguments passed into command line, where argv[0] is the name of the script itslef, https://stackoverflow.com/questions/29045768/how-to-use-sys-argv-in-python-to-check-length-of-arguments-so-it-can-run-as-scri
        # what this method is doing is, if the length of the arguments passed
        # in is two, this means what was passed in the command line was the
        # script name = argv[0] and the file with the args = argv[1]. This file
        # is 'args_xxxx_xxxx.txt' or similar
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    #args = parser.parse_args()
    args.gpu = int(args.gpu) if args.gpu is not None else 0
    args.distributed = False
    device = torch.device('cuda')
    # the actual PyTorch dataloader class is stored within self.data inside
    # the DepthDataLoader class, so we have to go through depth data loader
    # class before getting to data laoder class
    test = DepthDataLoader(args, 'online_eval').data
    # what this allows is defining different dataloaders with different
    # inherent properties depending on if it's loading testing data or
    # training data
    model = UnetAdaptiveBins.build(
        n_bins=args.n_bins,
        min_val=args.min_depth,
        max_val=args.max_depth,
        norm='linear').to(device)
    # after 'model_io.load_checkpoint(args.checkpoint_path, model)' is
    # executed, we only need the first element in the array which is the model
    model = model_io.load_checkpoint(args.checkpoint_path, model)[0]
    model = model.eval()        # configures the model to be in eval mode so turns off certain layers like batch norm, drop out, etc https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch

    #eval(model, test, args, gpus=['cuda'])
    eval(model, test, args, gpus=None)
