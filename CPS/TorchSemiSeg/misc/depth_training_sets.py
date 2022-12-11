import os
import shutil
import random

images = '/home/extraspace/Datasets/Datasets/cityscapes/leftImg8bit_trainvaltest (blurred)/leftImg8bit/'
train_extra = '/home/extraspace/Datasets/Datasets/cityscapes/leftImg8bit_trainvaltest (blurred)/leftImg8bit/train_extra/'
cityscapes_train = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/AdaBins/AdaBins/train_test_inputs/cityscapes_train_files_with_gt.txt'
cityscapes_test = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/AdaBins/AdaBins/train_test_inputs/cityscapes_test_files_with_gt.txt'
cityscapes_infer = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/AdaBins/AdaBins/train_test_inputs/cityscapes_inference_files_with_gt.txt'

t_list = []
v_list = []
i_list = []


# Creating Training and Validation Files for training the MDE on
# Cityscapes improved disparity depth
'''
for f in os.listdir(train_extra):
    city = os.path.join(train_extra, f)
    names = os.listdir(city)
    l = len(names)
    for i, pic in enumerate(names):
        if i < 0.97*l:
            t_list.append('/' + f + '/' + pic + ' /' + f + '/' + pic.split('.')[0] + '.png')
        else:
            v_list.append('/' + f + '/' + pic + ' /' + f + '/' + pic.split('.')[0] + '.png')

random.shuffle(t_list)
random.shuffle(v_list)

with open(cityscapes_train, 'w') as t, open(cityscapes_test, 'w') as v:
    t_len = len(t_list)
    v_len = len(v_list)
    for i, n in enumerate(t_list):
        t.write(n)
        if i != t_len-1:
            t.write('\n')
    for i, m in enumerate(v_list):
        v.write(m)
        if i) != v_len-1:
            v.write('\n')
'''

# Creating Inference Files for training the SemiSeg on Cityscapes improved
# disparity depth

for fl in os.listdir(images):
    if fl == 'train_extra':
        continue
    pth = os.path.join(images, fl)
    for f in os.listdir(pth):
        city = os.path.join(pth, f)
        for pic in os.listdir(city):
            i_list.append('/' + f + '/' + pic + ' None')

random.shuffle(i_list)

with open(cityscapes_infer, 'w') as i:
    i_len = len(i_list)
    for idx, im in enumerate(i_list):
        i.write(im)
        if idx != i_len - 1:
            i.write('\n')
