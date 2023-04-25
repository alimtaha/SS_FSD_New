import os, sys
import numpy as np
from matplotlib import pyplot as plt
import cv2
import glob
from tqdm import tqdm
from random import shuffle


path_1_16_labelled = '/home/extraspace/Datasets/Datasets/cityscapes/city/config_new/subset_train/train_aug_labeled_1-16.txt'        #186
path_1_16_unlabelled = '/home/extraspace/Datasets/Datasets/cityscapes/city/config_new/subset_train/train_aug_unlabeled_1-16.txt'    #2779
path_1_64_labelled = '/home/extraspace/Datasets/Datasets/cityscapes/city/config_new/subset_train/train_aug_labeled_1-64.txt'        #46
path_1_64_unlabelled = '/home/extraspace/Datasets/Datasets/cityscapes/city/config_new/subset_train/train_aug_unlabeled_1-64.txt'    #2929
path_1_128_labelled = '/home/extraspace/Datasets/Datasets/cityscapes/city/config_new/subset_train/train_aug_labeled_1-128.txt'      #23
path_1_128_unlabelled = '/home/extraspace/Datasets/Datasets/cityscapes/city/config_new/subset_train/train_aug_unlabeled_1-128.txt'  #2952

#Create 64 file, then use that to create 128 file

rng = np.random.default_rng()
idxs_64 = rng.choice(186, (46), replace=False).tolist()
idxs_128 = rng.choice(46, (23), replace=False).tolist()

print(len(idxs_64), len(idxs_128), idxs_128)

with open(path_1_64_labelled, 'w') as l_64:
    
    #Generating 1/64 Labels
    with open(path_1_16_labelled ,'r') as l_16:
        labels_16 = l_16.readlines()
    unlabelled_64 = labels_16.copy()
    labels_64 = []
    print(len(labels_16))
    for i in idxs_64:
        print(i, len(labels_16))
        labels_64.append(labels_16[i])
        unlabelled_64.remove(labels_16[i])
    print(len(unlabelled_64), 'Should be 140')
    with open(path_1_64_unlabelled, 'w') as un_64:
        with open(path_1_16_unlabelled, 'r') as un_16:
            unlabelled_16 = un_16.readlines()
            unlabelled_64 = unlabelled_16 + unlabelled_64
            print(len(unlabelled_64), 'Should be 2929')
            un_64.writelines(unlabelled_64)
    l_64.writelines(labels_64)
    
    #Generating 1/128 Labels
    unlabelled_128 = labels_64.copy()
    labels_128 = []
    for i in idxs_128:
        labels_128.append(labels_64[i])
        unlabelled_128.remove(labels_64[i])
    print(len(unlabelled_128), 'Should be 23', len(labels_128), 'Should be 23')   
    with open(path_1_128_labelled, 'w') as l_128:
        l_128.writelines(labels_128)
        unlabelled_128 = unlabelled_64 + unlabelled_128
        print(len(unlabelled_128), 'Should be 2952')
    with open(path_1_128_unlabelled, 'w') as un_128:
        un_128.writelines(unlabelled_128)
