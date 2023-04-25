#!/usr/bin/env python3
# encoding: utf-8
from dataloader_depth_append import ValPre
from network_depth_append import Network
from dataloader_depth_append import CityScape
from furnace.seg_opr.metric import hist_info, compute_score
from furnace.engine.logger import get_logger
from furnace.engine.evaluator import Evaluator
from furnace.utils.visualize import print_iou, show_img
from furnace.utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from config import config
import torch.multiprocessing as mp
import torch.nn as nn
import torch
import os
import cv2
import argparse
import numpy as np
import sys
sys.path.append('../../')


'''
try:
    from azureml.core import Run
    azure = True
    run = Run.get_context()
except:
    azure = False
'''
logger = get_logger()


class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        name = data['fn']
        #pred = self.sliding_eval(img, config.eval_crop_size, config.eval_stride_rate, device)
        print(img.mean)
        pred = self.whole_eval(img, config.eval_crop_size, device)

        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes,
                                                       pred,
                                                       label)
        print(hist_tmp.shape, 'hist.tmp shape')
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp,
                        'correct': correct_tmp}

        if self.save_path is not None:
            fn = name + '.png'
            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info('Save the image ' + fn)

        if self.show_image:
            colors = self.dataset.get_class_colors
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        print('hist shape', hist.shape)
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        print('hist shape in compute metric', hist)
        iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct,
                                                       labeled)
        # changed from the variable dataset to the class directly so this
        # function can now be called without first initialising the eval file
        print(len(dataset.get_class_names()))
        result_line = print_iou(iu, mean_pixel_acc,
                                dataset.get_class_names(), True)
        '''
        if azure:
            mean_IU = np.nanmean(iu)*100
            run.log(name='Test/Val-mIoU', value=mean_IU)
        '''
        return result_line


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = Network(
        config.num_classes,
        criterion=None,
        norm_layer=nn.BatchNorm2d)
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
    val_pre = ValPre()
    dataset = CityScape(data_setting, 'val', val_pre)

    with torch.no_grad():
        segmentor = SegEvaluator(
            dataset,
            config.num_classes,
            config.image_mean,
            config.image_std,
            network,
            config.eval_scale_array,
            config.eval_flip,
            all_dev,
            args.verbose,
            args.save_path,
            args.show_image)
        segmentor.run(config.snapshot_dir, args.epochs, config.val_log_file,
                      config.link_val_log_file)
