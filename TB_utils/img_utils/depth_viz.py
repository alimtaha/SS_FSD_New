import numpy as np 
from matplotlib import pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
import os
#from patchify import patchify, unpatchify

#Parameters

depth_max_viz = 80 #Max range for depth maps colorbars
error_vmax = 5 #Max range for error colorbar viz
save = False
plt_errors = True
save_path = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/SS_FSD_New/TB_utils/saved_assets/depth'
frame_name = 'depth_' + 'errors_' if plt_errors is True else ''
pics = ['frankfurt_000001_014406']

#TODO Need to mask out all errors in areas in the GT where the depth is over 80 if evaluating pictures with a max depth of 80

for pic in pics:

    imga = Image.open(f'/home/extraspace/Datasets/Datasets/cityscapes/city/images/val/{pic}.jpg')

    paths = [
        f'/home/extraspace/Logs/MDE/NewCRFs/0311_1343maxdepth:80.0_width:352_height:704_lr:4e-05_/summaries/epoch_14/{pic}.png',
        f'/home/extraspace/Logs/MDE/PixelFormer/0218_1909maxdepth:80.0_width:352_height:704_lr:4e-05_/summaries/epoch_19/{pic}.png',
        f'/home/extraspace/Logs/MDE/PixelFormer/0216_2312maxdepth:256.0_width:352_height:704_lr:4e-05_/summaries/epoch_19/{pic}.png',
        #f'/home/extraspace/Logs/MDE/PixelFormer/0208_1152maxdepth:80.0_width:352_height:704_lr:4e-05_/summaries/epoch_10/{pic}.png'
    ]

    city = pic.split('_')[0]
    gt = Image.open(f'/home/extraspace/Datasets/Datasets/cityscapes/depth/multi_new_depth_inf/val/{city}/{pic}.png')


    fig = plt.figure(figsize=(20, 15))

    if plt_errors is True:
        n_rows = len(paths) + 1
    else:
        n_rows = len(paths)/2 + 1

    fig.add_subplot(n_rows, 2, 1)
    plt.imshow(imga)

    fig.add_subplot(n_rows, 2, 2)
    gt = np.array(gt, dtype=np.float32)/256
    plt.imshow(gt, cmap='magma_r', vmin=0, vmax=depth_max_viz)
    plt.colorbar()

    for i in range(len(paths)):
        
        paths[i] = np.array(Image.open(paths[i]), dtype=np.float32)/256
        
        if plt_errors is True:
            #error calculations and plt
            plt_idx = 2*i + 3
            fig.add_subplot(n_rows, 2, plt_idx+1)
            plt.imshow((paths[i] - gt), cmap='magma_r', vmin=0, vmax=error_vmax)
            plt.colorbar()
        else:
            plt_idx = i + 3

        #depth map plot
        fig.add_subplot(n_rows, 2, plt_idx)
        plt.imshow(paths[i], cmap='magma_r', vmin=0, vmax=depth_max_viz)
        plt.colorbar()


    if save is True:
        print('saving')
        save_path = os.path.join(save_path, frame_name + pic + '.png')
        fig.savefig(save_path, dpi='figure')

    plt.show()
