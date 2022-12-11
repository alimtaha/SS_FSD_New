import numpy as np 
from matplotlib import pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from patchify import patchify, unpatchify

pics = '000000_002963'
imga = Image.open(f'/home/extraspace/Datasets/Datasets/cityscapes/city/images/val/frankfurt_{pics}.jpg')
label0 = Image.open(f'/home/extraspace/Datasets/Datasets/cityscapes/city/segmentation_weak_0/val/frankfurt/frankfurt_{pics}.png')
label1 = Image.open(f'/home/extraspace/Datasets/Datasets/cityscapes/city/segmentation_weak_0.1/val/frankfurt/frankfurt_{pics}.png')
label2 = Image.open(f'/home/extraspace/Datasets/Datasets/cityscapes/city/segmentation_weak_0.2/val/frankfurt/frankfurt_{pics}.png')
label3 = Image.open(f'/home/extraspace/Datasets/Datasets/cityscapes/city/segmentation_weak_0.3/val/frankfurt/frankfurt_{pics}.png')
label4 = Image.open(f'/home/extraspace/Datasets/Datasets/cityscapes/city/segmentation_weak_0.4/val/frankfurt/frankfurt_{pics}.png')
label5 = Image.open(f'/home/extraspace/Datasets/Datasets/cityscapes/city/segmentation_weak_0.5/val/frankfurt/frankfurt_{pics}.png')
gt = Image.open(f'/home/extraspace/Datasets/Datasets/cityscapes/city/segmentation/val/frankfurt_{pics}_gtFine.png')
#gt = np.array(gt, dtype = np.uint8)

#gt[gt>0] = 1


transt = transforms.ToTensor()
transp = transforms.ToPILImage()
#img = transt(imga).data[:,:1024,:1024]
#print(img.shape)
#patches_1 = img.unfold(0, 3, 3)
#patches_2 = patches_1.unfold(1, 256, 256)
#patches = patches_2.unfold(2, 256, 256)

patchify_img = transt(np.array(imga, dtype=np.float32).transpose(2,0,1)[:,:1024,:1024]).data
print(patchify_img.shape)

patches_img = patchify(patchify_img, (3,256,256),step=256)

print(patches_img.shape)

fig = plt.figure(figsize=(4, 4))
for i in range(4):
    for j in range(4):
        inter_patch = (patches_img[0][i][j]).transpose(1,2,0).astype(np.uint8)
        print(inter_patch.shape)
        inp = transp(inter_patch)
        inp = np.array(inp)

        ax = fig.add_subplot(4, 4, ((i*4)+j)+1, xticks=[], yticks=[])
        plt.imshow(inp)


recon = transp(unpatchify(patches_img, (3,1024,1024)).transpose(1,2,0).astype(np.uint8))
fig = plt.figure(figsize=(4, 4))
plt.imshow(recon)

plt.show()


#def reconstruct(patches):

'''
tensor = torch.tensor()

    for i in range(patches[0].shape):
        for i in range(patches[0].shape):


plt.figure(figsize=(10,10))
new_img = transp(fold3(fold2(fold1(patches))))
plt.imshow(new_img)


'''

'''
fig = plt.figure(figsize=(20, 15))
fig.add_subplot(2, 2, 1)
plt.imshow(gt, cmap='magma_r')

#fig.add_subplot(4, 2, 3)
#plt.imshow(label0, cmap='magma_r')

fig.add_subplot(2, 2, 3)
plt.imshow(label1, cmap='magma_r')

fig.add_subplot(2, 2, 4)
plt.imshow(label2, cmap='magma_r')

fig.add_subplot(2, 2, 2)
plt.imshow(img, cmap='magma_r')

#fig.add_subplot(4, 2, 4)
#plt.imshow(label3, cmap='magma_r')
#
#fig.add_subplot(4, 2, 6)
#plt.imshow(label4, cmap='magma_r')
#
#fig.add_subplot(4, 2, 8)
#plt.imshow(label5, cmap='magma_r')

plt.show()
'''