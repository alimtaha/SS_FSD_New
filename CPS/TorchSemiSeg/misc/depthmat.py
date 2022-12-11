from math import inf
import scipy.io as sio
import numpy as np
from PIL import Image
from torch import float32
import matplotlib.pyplot as plt
import cv2
import os
import path
import imageio
#import h5py

depth_path = "/home/extraspace/Datasets/Datasets/cityscapes/depth/depth_stereoscopic"
save_path = '/home/extraspace/Datasets/Datasets/cityscapes/depth/depth_png'
save = None
# test =

for m in os.listdir(depth_path):
    m_path = os.path.join(depth_path, m)
    for n in os.listdir(m_path):
        n_path = os.path.join(m_path, n)
        for o in os.listdir(n_path):
            pth = os.path.join(n_path, o)
            # print(pth)
            mat_content = sio.loadmat(pth)
            root = os.path.join(os.path.join(save_path, m), n)
            save = os.path.join(
                root, ('_').join(
                    o.split('.')[0].split('_')[
                        :3]) + '.png')
            # print(save)
            if os.path.exists(root) is False:
                os.makedirs(root)
            d_array = (mat_content['depth_map'].astype('float32')) * 256
            d_array[np.isinf(d_array)] = 65535
            d_array[np.where(d_array >= 65535)] = 65535
            # d_array.astype('uint16')
            imageio.imwrite(save, d_array.astype('uint16'))
            #im = Image.fromarray(d_array, 'I;16')
            # im.save(save)

'''
im_check = Image.open(test)
im_array = np.asarray(im_check, dtype=np.float32) / 256.0
im_array = im_array.astype('uint16')
plt.imshow(im_array, cmap='magma_r')
plt.colorbar()
plt.show()

mat_contents = sio.loadmat(depth_path)
print(mat_contents['depth_map'].shape, mat_contents['depth_map'].dtype)
matp = mat_contents['depth_map'].astype('float32')
print(matp.shape, matp.dtype)
matp = matp*256
#matp[np.where(matp>65535)] = 65535
matp[np.isinf(matp)] = 65535
matp[np.where(matp>=65535)] = 65535
matp = matp.astype('uint16')
print(matp.max())
laplacian = cv2.Laplacian(matp/256.0,cv2.CV_64F)
#cv2.imshow('Image', matp)
fig = plt.figure(figsize=(20, 14))
fig.add_subplot(2, 1, 1)
plt.imshow(matp/256.0, cmap='magma_r')
#cv2.waitKey(0)
plt.colorbar()
fig.add_subplot(2, 1, 2)


im1 = Image.fromarray(matp, 'I;16')
im = np.asarray(im1, dtype=np.float32) / 256.0
im = im.astype('uint16')
plt.imshow(im, cmap='magma_r')
plt.colorbar()
plt.show()

#im2 = Image.open(im1)

'''
