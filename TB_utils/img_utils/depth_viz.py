import numpy as np 
from matplotlib import pyplot as plt
from PIL import Image
import os
#from patchify import patchify, unpatchify

#Parameters

depth_max_viz = 80 #Max range for depth maps colorbars
error_vmax = 5 #Max range for error colorbar viz
save = True
plt_errors = True
save_path = os.getcwd()
frame_name = 'depth_' + 'errors_' if plt_errors is True else ''
pics = ['frankfurt_000001_014406']

#TODO Need to mask out all errors in areas in the GT where the depth is over 80 if evaluating pictures with a max depth of 80

for pic in pics:

    imga = Image.open(f'/mnt/Dataset/city/images/val/{pic}.jpg')

    root_path = '/mnt/Dataset/city/depth_gen/'

    root_path = '/home/extraspace/Datasets/Datasets/cityscapes/city/depth_gen/'

    path_names = {
        'n256': f'newcrfs_256.0_model-89964-best_silog_14.55338/{pic}.png',
        'n80': f'newcrfs_80.0_model-44982-best_d1_0.95066/{pic}.png',
        'p256': f'pixelformer_256.0_model-9996-best_d1_0.93959/{pic}.png',
        'p80': f'pixelformer_80.0_model-49980-best_d1_0.94092/{pic}.png'
    }

    city = pic.split('_')[0]
    gt256 = Image.open(f'/home/extraspace/Datasets/Datasets/cityscapes/depth/multi_new_depth_inf/val/{city}/{pic}.png')
    #gt80 = Image.open(f'/home/extraspace/Datasets/Datasets/cityscapes/depth/depth_80/val/{city}/{pic}.png')


    fig = plt.figure(figsize=(20, 15))

    if plt_errors is True:
        n_rows = len(path_names) + 1
    else:
        n_rows = len(path_names)/2 + 1

    fig.add_subplot(n_rows, 2, 1)
    plt.imshow(imga)

    fig.add_subplot(n_rows, 2, 2)
    gt256 = np.array(gt256, dtype=np.float32)/256
    gt80 = np.where(gt256>80, 0, gt256)
    plt.imshow(gt80, cmap='magma_r', vmin=0, vmax=depth_max_viz)
    plt.colorbar()

    paths = []

    for i, k in enumerate(path_names):
        
        pth = os.path.join(root_path, path_names[k])
        paths.append(np.array(Image.open(pth), dtype=np.float32)/256)
        
        if plt_errors is True:
            #error calculations and plt
            plt_idx = 2*i + 3
            fig.add_subplot(n_rows, 2, plt_idx+1)
            plt.title(path_names[k])
            gt = gt256 if k.endswith('6') else gt80
            plt.imshow((paths[i] - gt256), cmap='magma_r', vmin=0, vmax=error_vmax)
            plt.colorbar()
        else:
            plt_idx = i + 3

        #depth map plot
        fig.add_subplot(n_rows, 2, plt_idx)
        plt.title(path_names[k])
        plt.imshow(paths[i], cmap='magma_r', vmin=0, vmax=depth_max_viz)
        plt.colorbar()


    if save is True:
        print('saving')
        save_path = os.path.join(save_path, 'depth_gen' + frame_name + pic + '.png')
        fig.savefig(save_path, dpi='figure')

    plt.show()
