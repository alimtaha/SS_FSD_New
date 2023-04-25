import numpy as np 
from matplotlib import pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
#from patchify import patchify, unpatchify

path256_16 = '/home/extraspace/Logs/MDE/PixelFormer/0206_1559maxdepth:256.0_width:352_height:704_lr:4e-05_/summaries/epoch_16/frankfurt_000001_014406.png'
path256_10 = '/home/extraspace/Logs/MDE/PixelFormer/0206_1559maxdepth:256.0_width:352_height:704_lr:4e-05_/summaries/epoch_10/frankfurt_000001_014406.png'
path80_14 = '/home/extraspace/Logs/MDE/PixelFormer/0208_1152maxdepth:80.0_width:352_height:704_lr:4e-05_/summaries/epoch_13/frankfurt_000001_014406.png'
path80_10 = '/home/extraspace/Logs/MDE/PixelFormer/0208_1152maxdepth:80.0_width:352_height:704_lr:4e-05_/summaries/epoch_10/frankfurt_000001_014406.png'

pics = '000001_014406'
imga = Image.open(f'/home/extraspace/Datasets/Datasets/cityscapes/city/images/val/frankfurt_{pics}.jpg')
#label0 = Image.open(f'/home/extraspace/Datasets/Datasets/cityscapes/city/segmentation_weak_0/val/frankfurt/frankfurt_{pics}.png')
path256_16 = Image.open(path256_16)
path256_10 = Image.open(path256_10)
path80_13 = Image.open(path80_14)
path80_10 = Image.open(path80_10)
#label5 = Image.open(f'/home/extraspace/Datasets/Datasets/cityscapes/city/segmentation_weak_0.5/val/frankfurt/frankfurt_{pics}.png')
gt = Image.open(f'/home/extraspace/Datasets/Datasets/cityscapes/depth/multi_new_depth_inf/val/frankfurt/frankfurt_000001_014406.png')
#gt = np.array(gt, dtype = np.uint8)

#gt[gt>0] = 1


transt = transforms.ToTensor()
transp = transforms.ToPILImage()
#img = transt(imga).data[:,:1024,:1024]
#print(img.shape)
#patches_1 = img.unfold(0, 3, 3)
#patches_2 = patches_1.unfold(1, 256, 256)
#patches = patches_2.unfold(2, 256, 256)

#patchify_img = transt(np.array(imga, dtype=np.float32).transpose(2,0,1)[:,:1024,:1024]).data
#print(patchify_img.shape)

#patches_img = patchify(patchify_img, (3,256,256),step=256)

#print(patches_img.shape)

# fig = plt.figure(figsize=(4, 4))
# for i in range(4):
#     for j in range(4):
#         inter_patch = (patches_img[0][i][j]).transpose(1,2,0).astype(np.uint8)
#         print(inter_patch.shape)
#         inp = transp(inter_patch)
#         inp = np.array(inp)

#         ax = fig.add_subplot(4, 4, ((i*4)+j)+1, xticks=[], yticks=[])
#         plt.imshow(inp)


# recon = transp(unpatchify(patches_img, (3,1024,1024)).transpose(1,2,0).astype(np.uint8))
# fig = plt.figure(figsize=(4, 4))
# plt.imshow(recon)

# plt.show()


#def reconstruct(patches):

'''
tensor = torch.tensor()

    for i in range(patches[0].shape):
        for i in range(patches[0].shape):


plt.figure(figsize=(10,10))
new_img = transp(fold3(fold2(fold1(patches))))
plt.imshow(new_img)


'''


fig = plt.figure(figsize=(20, 15))

fig.add_subplot(6, 2, 1)
plt.imshow(imga)#, cmap='magma_r')

#fig.add_subplot(4, 2, 3)
#plt.imshow(label0, cmap='magma_r')

fig.add_subplot(5, 2, 2)
gt = np.array(gt, dtype=np.float32)/256
plt.imshow(gt, cmap='magma_r', vmin=0, vmax=80)
plt.colorbar()

fig.add_subplot(5, 2, 3)
path256_16 = np.array(path256_16, dtype=np.float32)/256
plt.imshow(path256_16, cmap='magma_r', vmin=0, vmax=80)
plt.colorbar()

fig.add_subplot(5, 2, 4)
path256_16_error = path256_16 - gt
plt.imshow(path256_16_error, cmap='magma_r', vmin=0, vmax=5)
plt.colorbar()

fig.add_subplot(5, 2, 5)
path256_10 = np.array(path256_10, dtype=np.float32)/256
plt.imshow(path256_10, cmap='magma_r', vmin=0, vmax=80)
plt.colorbar()

fig.add_subplot(5, 2, 6)
path256_10_error = path256_10 - gt
plt.imshow(path256_10_error, cmap='magma_r', vmin=0, vmax=5)
plt.colorbar()

fig.add_subplot(5, 2, 7)
path80_13 = np.array(path80_13, dtype=np.float32)/256
plt.imshow(path80_13, cmap='magma_r', vmin=0, vmax=80)
plt.colorbar()
#

fig.add_subplot(5, 2, 8)
path80_13_error = path80_13 - gt
plt.imshow(path80_13_error, cmap='magma_r', vmin=0, vmax=5)
plt.colorbar()

fig.add_subplot(5, 2, 9)
path80_10 = np.array(path80_10, dtype=np.float32)/256
plt.imshow(path80_10, cmap='magma_r', vmin=0, vmax=80)
plt.colorbar()

fig.add_subplot(5, 2, 10)
path80_10_error = path80_10 - gt
plt.imshow(path80_10_error, cmap='magma_r', vmin=0, vmax=5)
plt.colorbar()

#
#fig.add_subplot(4, 2, 8)
#plt.imshow(label5, cmap='magma_r')

plt.show()
