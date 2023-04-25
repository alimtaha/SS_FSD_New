import os, sys
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
from tqdm import tqdm
from random import shuffle

paths = [
    '/home/extraspace/Datasets/Datasets/cityscapes/leftImg8bit_trainvaltest_blurred/leftImg8bit/train/**/*.jpg',
    '/home/extraspace/Datasets/Datasets/cityscapes/leftImg8bit_trainvaltest_blurred/leftImg8bit/val/**/*.jpg',
    '/home/extraspace/Datasets/Datasets/cityscapes/leftImg8bit_trainvaltest_blurred/leftImg8bit/test/**/*.jpg',
    ]
         

train_file = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/SS_FSD_New/PixelFormer/data_splits/cityscapes_all_edited.txt'

image_paths = []
for p in paths:
    image_paths = image_paths + glob.glob(p, recursive=True)
    
print(len(image_paths))
shuffle(image_paths)

with open(train_file, 'w') as f:
    for n in image_paths:
        f.write(('/').join(n.split('/')[-3:]) + ' None None\n')