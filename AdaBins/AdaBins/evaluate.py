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


def compute_errors(gt, pred):
    # for all indices, the maximum of the gt/pred or pred/gt is computed
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()  # mean of all thresholds less than 1.25
    a2 = (thresh < 1.25 ** 2).mean()  # mean of all thresholds less than 1.25^2
    a3 = (thresh < 1.25 ** 3).mean()  # mean of all thresholds less than 1.25^3

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(
        a1=a1,
        a2=a2,
        a3=a3,
        abs_rel=abs_rel,
        rmse=rmse,
        log_10=log_10,
        rmse_log=rmse_log,
        silog=silog,
        sq_rel=sq_rel)


# def denormalize(x, device='cpu'):
#     mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
#     std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
#     return x * std + mean
#
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

    metrics = RunningAverageDict()
    # crop_size = (471 - 45, 601 - 41)
    # bins = utils.get_bins(100)
    total_invalid = 0
    plotting_factor = 0
    with torch.no_grad():  # reducing gradient computation during inference will allow faster running
        model.eval()  # configuring model to be in eval mode

        sequential = test_loader
        for batch in tqdm(
                sequential):  # dataloader object returns batches of data

            # print(batch)
            #print("batch size")
            # print(len(batch))
            image = batch['image'].to(torch.device('cuda'))
            #print('image dimensions', image.shape)
            gt = batch['depth'].to(torch.device('cuda'))
            #print('gt dimensions', gt.shape)
            # only returns the pred image
            final = predict_tta(model, image, args)
            #print('pred output (from pred_tta) before squeeze', final.shape)
            # the image is squeezed into a (1?) dim array for computing all
            # erros
            final = final.squeeze().cpu().numpy()
            #print('pred output (from pred_tta) after squeeze', final.shape)

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
                    dpath = batch['image_path'][0].split(
                        '/')  # depth path derived
                    impath = dpath[1] + "_" + dpath[-1]
                    impath = impath.split('.')[0]  # image path derived
                    factor = 256

                #rgb_path = os.path.join(rgb_dir, f"{impath}.png")
                # tf.ToPILImage()(denormalize(image.squeeze().unsqueeze(0).cpu()).squeeze()).save(rgb_path)
                # path set for saving pred output
                pred_path = os.path.join(args.save_dir, f'{impath}.png')
                print(pred_path)
                # converting them back to a format where they can be compared
                # with the depth maps (so converting the model output from
                # metres to 16bit unsigned int where every bit is 256th of a
                # metre)
                pred = (final * factor).astype('uint16')
                # converts array to a PIL image and saves in pred path
                Image.fromarray(pred).save(pred_path)

            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    # print("Invalid ground truth")
                    total_invalid += 1
                    continue

            print("gt dimensions from eval before squeeze", gt.shape)
            # batch has multiple images, the entire batch is squeezed into a
            # (1?) dim array for computing all erros
            gt = gt.squeeze().cpu().numpy()
            # .numpy() method is pretty much straightforward. It converts a tensor object into an numpy.ndarray object. This implicitly means that the converted tensor will be now processed on the CPU.
            print("gt dimensions from eval after squeeze", gt.shape)
            # np.logical_and is generates a true if both conditions within
            # brackets are true otherwise false
            valid_mask = np.logical_and(
                gt > args.min_depth, gt < args.max_depth)
            # so if the element_wise value at gt is both greater than min depth and less than the max depth it a true is output
            # size of the output is equal to gt
            # imp to note that the valid_mask sets all invalid depth values
            # (anything less than min or greater than 80m) to 0, this way,
            # these values are not taken into account for the network
            # evaluation and error computation metrics
            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt.shape
                # getting shape of eval mask from the logical_and operation
                # above and filling it with zeros
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[int(0.40810811 *
                                  gt_height):int(0.99189189 *
                                                 gt_height), int(0.03594771 *
                                                                 gt_width):int(0.96405229 *
                                                                               gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),  # this is the eigen crop, crops images from 1216 x 352 to 704 x 352, everything outside the eigen crop is set to 0 in the mask
                                  int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1  # setting certain indices in the eval mask array to 1, everythinhg outside is zeros as per line 132
                    else:  # the 0.0359477 and 0.96405229 will correspond to an integer when multiplied by the size of the image, typesetting with int will ensure its a whole number
                        eval_mask[45:471, 41:601] = 1
            # so to sum up, in line 127 all invalid (out of threshold depth
            # values are masked)
            valid_mask = np.logical_and(valid_mask, eval_mask)
            #             gt = gt[valid_mask]                                           #then following that, only the indices required using eval mask are kept at ones
            # final = final[valid_mask]                                     #so
            # in the end, all out of depth threshold or out of index values are
            # kept to 0, else keep at 1

            '''
            #if plotting_factor == 25: #only print the 25th image in the evaluation files
            fig = plt.figure(figsize=(10, 7))
            fig.add_subplot(3, 1, 1)
            sampleio = Image.open(os.path.join(args.data_path_eval, batch['image_path'][0]))

            plt.imshow(sampleio)
            fig.add_subplot(3, 1, 2)
            plt.imshow(final, cmap='hsv')
            plt.colorbar()
            fig.add_subplot(3, 1, 3)
            plt.imshow(gt, cmap='hsv')
            plt.colorbar()
            plt.show()
            #else:
            #    plotting_factor=plotting_factor+1

            #print("plotting_factor", plotting_factor)
            '''
            metrics.update(
                compute_errors(
                    gt[valid_mask],
                    final[valid_mask]))  # even tho the size of the tensors being passed in are 1216 x 352, the use of the mask means they don't have any effect on the calculation of the errors in the compute_errors function

    print(f"Total invalid: {total_invalid}")
    metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
    print(f"Metrics: {metrics}")


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
