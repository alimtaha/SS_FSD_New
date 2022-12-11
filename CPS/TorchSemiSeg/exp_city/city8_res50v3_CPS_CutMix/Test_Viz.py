from PIL import Image
import torch.nn as nn
from torchvision.transforms import ToTensor
import numpy as np
import cv2
import sys
import os
sys.path.append('../../')


'''label = Image.open(
    "/home/extraspace/Datasets/Datasets/cityscapes/city/segmentation/train/aachen_000000_000019_gtFine.png")
label = np.asarray(label, dtype=np.uint8)
img = Image.open(
    "/home/extraspace/Datasets/Datasets/cityscapes/city/images/train/aachen_000000_000019.jpg")

T = ToTensor()

img = T(img)
img = img[:3,:100,:100]
print(img.shape)
img_v = img.contiguous().view(3,-1)
print(img_v.shape)
print(img.shape, img[1,99,0], img_v[1,9900])
'''

save_path = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/CPS/TorchSemiSeg/exp_city/city8_res50v3_CPS_CutMix'
v3_save = os.path.join(save_path, 'v3_array')
feats_save = os.path.join(save_path, 'feats_array')
#v3 - 1024 * 2048 = 2097152
v3_feats_idx = np.arange(2097152)
v3_feats_idx = np.random.choice(v3_feats_idx, 2000, replace=False)
np.save(v3_save, v3_feats_idx)

feats_quantile = np.arange(40000)
feats_idx = np.random.choice(feats_quantile, 1000, replace=False)
feats_idx = np.concatenate((feats_idx, 221072+feats_idx))
print(feats_idx.shape)
np.save(feats_save, feats_idx)