import numpy as np
import torch
from PIL import Image
import os
from matplotlib import pyplot as plt
import cv2

with open('../data_splits/cityscapes_val_edited.txt', 'r') as f:
    filenames = f.readlines()
sample_path = filenames[8]
print(sample_path)

semantic_path = os.path.join('/home/extraspace/Datasets/Datasets/cityscapes/city/segmentation/val', sample_path.split()[1].split('/')[-1].replace('.png', '_gtFine.png'))
print(semantic_path)

road_mask = Image.open(semantic_path)
#road_mask = cv2.imread(semantic_path, cv2.IMREAD_GRAYSCALE)
road_mask = np.asarray(road_mask)
print(road_mask[700,250])
# print('BEFORE', 20*'===', road_mask)
# plt.imshow(road_mask, cmap='magma_r')
# plt.colorbar()
# plt.savefig('test.png')
road_mask  = np.where(road_mask == 0, 1, 0)
# print('AFTER', 20*'+++', road_mask)
# road_mask = np.expand_dims(road_mask, axis=2)


# gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype
plt.imshow(road_mask, cmap='magma_r')
plt.colorbar()
plt.savefig('test.png')