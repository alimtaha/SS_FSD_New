import os, sys
import numpy as np
from matplotlib import pyplot as plt
import cv2
from numba import jit
import glob
from tqdm import tqdm

paths = {
    'KITTI Train': '/home/extraspace/Datasets/Datasets/kitti/depth/**/proj_depth/groundtruth/image_02/*.png',
    'Cityscapes Train 256': '/home/extraspace/Datasets/Datasets/cityscapes/depth/multi_new_depth_inf/train_extra/**/*.png',
    'Cityscapes Train 80': '/home/extraspace/Datasets/Datasets/cityscapes/depth/depth_80/train_extra/**/*.png',
    }

#@jit(nopython=True)
def calc_metrics_from_histogram (hist, bins):
    print('here')

    mid_points = 0.5*(bins[1:] + bins[:-1])
    mean = np.average(mid_points, weights=hist)
    non_zero_mean = np.average(mid_points[1:], weights=hist[1:])

    std = np.sqrt(np.average((mid_points - mean)**2, weights=hist))
    non_zero_std = np.sqrt(np.average((mid_points[1:] - non_zero_mean)**2, weights=hist[1:]))

    return (mean, std, non_zero_mean, non_zero_std)


#@jit(nopython=True)
def calc_histograms(image_paths, name, raw_array):
    if k[:10] == 'Cityscapes':
        rolling_mean = np.zeros((1024, 2048), dtype=np.float32)
        rolling_std = np.zeros((1024, 2048), dtype=np.float32)
    else:
        rolling_mean = np.zeros((352, 1216), dtype=np.float32)
        rolling_std = np.zeros((352, 1216), dtype=np.float32)

    max_range = 256 if name[-1] == '6' else 80

    for idx, pth in tqdm(enumerate(image_paths)):
        img = cv2.imread(pth, -1) / 256.0
        # img = img.astype(np.int8)
        
        if name[:10] != 'Cityscapes':
            
            #kb_crop to ensure consistent KITTI image sizes 352 X 1216
            height = img.shape[0]
            width = img.shape[1]
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            img = img[top_margin:top_margin + 352, left_margin:left_margin + 1216]
        
        if idx > 0:
            hist += np.histogram(img.flatten(), bins = max_range, range = (0, max_range))[0]
            hist_precise += np.histogram(img.flatten(), bins = 100*max_range, range = (0, max_range))[0]
        else:
            hist, bins = np.histogram(img.flatten(), bins = max_range, range = (0, max_range))
            hist_precise, bins_precise = np.histogram(img.flatten(), bins = 100*max_range, range = (0, max_range))

        rolling_mean += img
        #raw_array[:,:,idx] = img
    
    mean_image = rolling_mean / len(image_paths)

    #calc rolling std dev manually
    for idx, pth in tqdm(enumerate(image_paths)):
        img = cv2.imread(pth, -1) / 256.0
        
        if name[:10] != 'Cityscapes':
            
            #kb_crop to ensure consistent KITTI image sizes 352 X 1216
            height = img.shape[0]
            width = img.shape[1]
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            img = img[top_margin:top_margin + 352, left_margin:left_margin + 1216]
        
        rolling_std += (img - mean_image)**2
    
    std_image = np.sqrt(rolling_std / len(image_paths))

    # raw_nans = raw_array.flatten()
    # raw_nans[raw_array < 0.1] = np.nan
    # mean_nozeros = np.nanmean(raw_nans)
    # std_nozeros = np.nanstd(raw_nans)

    mean, std, non_zero_mean, non_zero_std = calc_metrics_from_histogram(hist_precise, bins_precise)
    #reshaped = raw_array.reshape(-1)
    # hist, bins = np.histogram(reshaped, bins = max_range, range = (0, max_range))
    sorted_idx = np.argsort(hist)[::-1]
    
    fig = plt.figure(figsize=(12, 12), dpi=250)
    plt.rcParams.update({'font.size': 16})

    plt.bar(bins[:-1], hist,  width=1, align='edge')
    plt.title(f'Depth Distribution    -     Max Bin Values {sorted_idx[0:5]}')
    plt.xlabel('Depth Pixel Value Bins')
    plt.ylabel('Count')
    plt.savefig(f'/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Desktop/new_distributions/{name}_histogram.png')
    
    plt.clf()
    plt.bar(bins[1:-1], hist[1:],  width=1, align='edge')
    plt.title(f'Depth Distribution (Zero Excluded)    -     Max Bin Values {sorted_idx[1:6]}')
    plt.xlabel('Depth Pixel Value Bins')
    plt.ylabel('Count')
    plt.savefig(f'/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Desktop/new_distributions/{name}_histogram_zeroexcluded.png')

    #mean = reshaped.mean()
    plt.clf()
    plt.imshow(mean_image, cmap='magma_r')
    plt.title(f'Mean Image    -     Mean: {round(mean, 2)}, Zero Excluded:{round(non_zero_mean, 2)}') 
    plt.colorbar()
    plt.savefig(f'/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Desktop/new_distributions/{name}_meanimg.png')

    #std_dev = reshaped.std()
    plt.clf()
    plt.imshow(std_image, cmap='magma_r')
    plt.title(f'Std Image    -     Std Dev: {round(std, 2)}, Zero Excluded: {round(non_zero_std, 2)}')
    plt.colorbar()
    plt.savefig(f'/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Desktop/new_distributions/{name}_stdimg.png')

if __name__ == '__main__':
    
    for k, v in paths.items():
        
        image_paths = glob.glob(v, recursive=True)
        n = len(image_paths)

        if k[:10] == 'Cityscapes':
            raw_array = np.zeros((1024, 2048, n), dtype=np.int8)
            im_array = np.zeros((1024, 2048), dtype = np.float32)
        else:
            raw_array = np.zeros((352, 1216, n), dtype=np.int8)
        
        calc_histograms(image_paths, k, raw_array)

