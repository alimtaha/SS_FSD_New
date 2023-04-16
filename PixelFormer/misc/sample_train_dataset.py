import os
import sys
import numpy
import random

full_dset = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/SS_FSD_New/PixelFormer/data_splits/cityscapes_train_extra_edited.txt'
sampled_dset = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/SS_FSD_New/PixelFormer/data_splits/cityscapes_train_extra_sampled.txt'

with open(full_dset, 'r') as f:
    full_array = f.readlines()

random.shuffle(full_array)

sampled_array = random.sample(full_array, 5000)

with open (sampled_dset, 'w') as s:
    s.writelines(sampled_array)