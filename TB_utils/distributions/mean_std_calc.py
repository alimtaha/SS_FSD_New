import os, sys
import numpy as np
from matplotlib import pyplot as plt
import cv2
from numba import jit
import glob
from tqdm import tqdm

'''
Historical Values (Compare with GT dataset depth distributions and mean/std values in thesis)

Model: newcrfs_256.0_model-89964-best_silog_14.55338
============================================================ /n
mean:  29.090656
std:  31.79855
============================================================ /n


Model: newcrfs_80.0_model-44982-best_d1_0.95066
============================================================ /n
mean:  24.187988
std:  19.529781
============================================================ /n

'''


paths = [
    '/home/extraspace/Datasets/Datasets/cityscapes/city/depth_gen/newcrfs_256.0_model-89964-best_silog_14.55338/*', 
    '/home/extraspace/Datasets/Datasets/cityscapes/city/depth_gen/newcrfs_80.0_model-44982-best_d1_0.95066/*'
    ]

#@jit(nopython=True)
def calc_histograms(image_paths, raw_array):
    for idx, pth in tqdm(enumerate(image_paths)):
        img = cv2.imread(pth, -1) / 256.0
        raw_array[:,:,idx] = img

    mean = raw_array.mean()
    std = raw_array.std()

    print(20*'===', '/n')
    print('mean: ', mean)
    print('std: ', std)
    print(20*'===', '/n')
        # img = img.astype(np.int8)



if __name__ == '__main__':
    
    for n in paths:
        
        image_paths = glob.glob(n, recursive=True)
        n = len(image_paths)
        print(n)

        raw_array = np.zeros((1024, 2048, n), dtype=np.float32)
        
        calc_histograms(image_paths, raw_array)

