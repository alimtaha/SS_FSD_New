import os
import shutil
from PIL import Image
import numpy as np

'Calc Mean and Std Dev of Depth Dataset for Standardisation'

sup_source = '/home/extraspace/Datasets/Datasets/cityscapes/city/config_new/subset_train/train_aug_labeled_1-8_depth.txt'
unsup_source = '/home/extraspace/Datasets/Datasets/cityscapes/city/config_new/subset_train/train_aug_unlabeled_1-8_depth.txt'
depth_source = '/home/extraspace/Datasets/Datasets/cityscapes/city/depth_adabins'

d_images = np.zeros((3475, 1024, 2048))
print(d_images.shape)

image_list = os.listdir(depth_source)

for n, line in enumerate(image_list):
    depth_line = depth_source + '/' + line
    img = Image.open(depth_line)
    d_img = np.asarray(img, dtype=np.float32)
    d_images[n] = d_img / 256.0

d = d_images.flatten()
mean = np.mean(d)
std = np.std(d)
#std_1 = d.std()


print(mean, '\n', std)
