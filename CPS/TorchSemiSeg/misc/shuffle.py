import os
import random

source = '/home/extraspace/Datasets/Datasets/cityscapes/city/config_new/subset_train/train_aug_unlabeled_1-16_shuffle.txt'

lines = open(source).readlines()
random.shuffle(lines)
open(source, 'w').writelines(lines)
