import cv2
from cv2 import INTER_NEAREST
from matplotlib import pyplot as plt
import numpy as np
import scipy

img = cv2.imread(
    '/home/extraspace/Datasets/Datasets/cityscapes/leftImg8bit_trainvaltest_(blurred)/leftImg8bit/train/bochum/bochum_000000_000600.jpg')
im2 = cv2.imread(
    '/home/extraspace/Datasets/Datasets/cityscapes/leftImg8bit_trainvaltest_(blurred)/leftImg8bit/train/bochum/bochum_000000_000600.jpg')


#im2 = cv2.pyrDown(img)

print(img.shape)

#im2 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)



im4 = cv2.blur(img, (24, 24))

im5 = cv2.GaussianBlur(img, (37, 37), 0)
im5 = cv2.resize(im5, dsize=(img.shape[1] // 16, img.shape[0] // 16))
im5 = cv2.resize(im5, dsize=(img.shape[1], img.shape[0]))



im2 = cv2.resize(im2, dsize=(img.shape[1] // 16, img.shape[0] // 16))
im2 = cv2.resize(im2, dsize=(img.shape[1], img.shape[0]))

im3 = cv2.resize(img, dsize=(img.shape[1] // 20, img.shape[0] // 20))
im3 = cv2.resize(im3, dsize=(img.shape[1], img.shape[0]))




h,w,c = img.shape

#fig = plt.figure(figsize=(20, 14))

factor = 4
masks = np.ones(h//factor*w//factor)
indices = np.arange(0,h//factor*w//factor)
indices = np.random.choice(indices, size=h//factor*w//factor//2, replace=False)
masks[indices] = 0

masks_reshaped = masks.reshape(h//factor,w//factor).astype(np.int)
masks_upsized = cv2.resize(masks_reshaped, dsize=(w, h), interpolation=INTER_NEAREST)
im2 = cv2.resize(img, dsize=(img.shape[1] // 8, img.shape[0] // 8))
im2 = cv2.resize(im2, dsize=(img.shape[1], img.shape[0]))
im2 = cv2.GaussianBlur(im2, (9, 9), 0)


img20 = im2 * masks_upsized[...,np.newaxis]

#fig.add_subplot(2, 1, 1)
#plt.imshow(img20)

pixel_factor = 8

color_channel = False
h, w, c = img.shape

if color_channel is False:
    c = 1

h_reduced = h // pixel_factor
w_reduced = w // pixel_factor

masks = np.ones(h_reduced * w_reduced * c)
indices = np.arange(0,h_reduced * w_reduced * c)
indices = np.random.choice(indices, size= c * h_reduced * w_reduced // 2, replace=False)

masks[indices] = 0
masks = masks.reshape(h_reduced, w_reduced, c).astype(np.int)
masks = cv2.resize(masks, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
print(masks.shape)

if color_channel is False:
    masks = masks[...,np.newaxis]

im2 = cv2.resize(img, dsize=(img.shape[1] // 8, img.shape[0] // 8))
im2 = cv2.resize(im2, dsize=(img.shape[1], img.shape[0]))
im2 = cv2.GaussianBlur(im2, (9, 9), 0)

img200 = im2 * masks

#fig.add_subplot(2, 1, 2)
plt.imshow(img200)
plt.show()

#print(choice.shape)

#choice = np.reshape(choice, img[...,:1].shape)
#print(choice.shape)


'''
#im3 = cv2.pyrDown(im2)
#im3 = cv2.resize(im3, dsize=(img.shape[1], img.shape[0]))

fig = plt.figure(figsize=(20, 14))
fig.add_subplot(4, 2, 1)        
plt.imshow(img)                 #original image
fig.add_subplot(4, 2, 3)
im2 = cv2.resize(im2, dsize=(img.shape[1] // 16, img.shape[0] // 16))
im2 = cv2.resize(im2, dsize=(img.shape[1], img.shape[0]))
plt.imshow(im2)                 #single downsample
fig.add_subplot(4, 2, 5)
im3 = cv2.resize(img, dsize=(img.shape[1] // 24, img.shape[0] // 24))
im3 = cv2.resize(im3, dsize=(img.shape[1], img.shape[0]))
plt.imshow(im3)                 #double downsample
fig.add_subplot(4, 2, 7)
im7 = cv2.resize(img, dsize=(img.shape[1] // 48, img.shape[0] // 48))
im7 = cv2.resize(im7, dsize=(img.shape[1], img.shape[0]))
plt.imshow(im7)                 #triple downsample
fig.add_subplot(2, 2, 1)        
im6 = cv2.blur(im2, (24, 24), 0)
plt.imshow(im6)                 #single downsample followed by blur
fig.add_subplot(2, 2, 3)
im9 = cv2.GaussianBlur(im2, (37, 37), 0)
plt.imshow(im9)                 #single downsample followed by Gaussian Blur
fig.add_subplot(2, 2, 2)
im8 = cv2.blur(im3, (24, 24), 0)
plt.imshow(im2)                 #double downsample followed by blur
fig.add_subplot(2, 2, 4)
im40 = cv2.GaussianBlur(im3, (31, 31), 0)
plt.imshow(im40)                 #double downsample followed by Gaussian Blur
plt.show()
'''