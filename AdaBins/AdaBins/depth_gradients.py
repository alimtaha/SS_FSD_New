import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

i = 2

str = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/AdaBins/AdaBins/predictions_cityscapes/'
str2 = '/home/extraspace/Datasets/Datasets/cityscapes/leftImg8bit_trainvaltest_(blurred)/leftImg8bit/val/frankfurt/'
str3 = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/NeWCRFs/models/newcrfs_cityscapes/summaries/0807_2230NewCRFs20Epochs_2e-05LR_704x352crop_256.0max_depth0.001min_depthFalse_Log(Manual)FalseDisparityFalse_CPU/epoch_11/'
str4 = '/home/extraspace/Datasets/Datasets/cityscapes/depth/multi_new_depth_inf/val/frankfurt/'
#str4 = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/NeWCRFs/models/result_newcrfs_cityscapes/raw/leftImg8bit_'

img_list = ['frankfurt_000000_000294.png', 'frankfurt_000000_000576.png',
            'frankfurt_000000_001016.png', 'frankfurt_000000_001236.png',
            'frankfurt_000000_009969.png', 'frankfurt_000001_003056.png']

fig = plt.figure(figsize=(20, 14))

for i, m in enumerate(img_list):
    print(m)
    img = Image.open((str+m))
    img2 = Image.open(str2+m.replace('.png', '.jpg'))
    img3 = Image.open(str3+m)
    img4 = Image.open(str4+m)
    fig.add_subplot(4, 6, i+1)
    plt.imshow(img2)
    test_img1 = np.asarray(img, dtype=np.float32) / 256.0
    fig.add_subplot(4, 6, 7+i)
    plt.imshow(test_img1, cmap='magma_r', vmin=0, vmax=80)
    #plt.colorbar()
    fig.add_subplot(4, 6, 13+i)
    test_img2 = np.asarray(img3, dtype=np.float32) / 256.0
    plt.imshow(test_img2, cmap='magma_r', vmin=0, vmax=80)
    #plt.colorbar()
    fig.add_subplot(4, 6, 19+i)
    test_img3 = np.asarray(img4, dtype=np.float32) / 256.0
    plt.imshow(test_img3, cmap='magma_r', vmin=0, vmax=80)
    #plt.colorbar()

    #   '/home/extraspace/Datasets/Datasets/cityscapes/disparity/val/frankfurt/frankfurt_000000_000294_disparity.png')
    #im5 = Image.open('/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/NeWCRFs/models/result_newcrfs_cityscapes/raw/leftImg8bit_frankfurt_000000_000294.png')
#img4 = Image.open('/home/extraspace/Datasets/Datasets/cityscapes/city/images/train/bochum_000000_010700.png')
#img5 = Image.open('/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/AdaBins/AdaBins/predictions_cityscapes/hamburg_000000_105724.png')
#img6 = Image.open('/home/extraspace/Datasets/Datasets/cityscapes/city/images/train/hamburg_000000_105724.png')
#img2 = cv.imread('/media/taha_a/T7/Datasets/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png', 0)
plt.show()


'''
laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=7)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=7)
plt.subplot(5,1,1),plt.imshow(img2,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(5,1,2),plt.imshow(img,cmap = 'magma_r')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(5,1,3),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(5,1,4),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(5,1,5),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
'''

disp = np.array(im5).astype(np.float)
disp[disp > 0] = (disp[disp > 0] - 1) / 256
depth = 0.222126 * 2268.36 / disp


#cv.imshow('Image', img)
#cv.imshow('Image', img2)
# cv.waitKey(0)

'''
cv2.imshow('Gradients_X',edges_x)
cv2.imshow('Gradients_Y',edges_y)
cv2.waitKey(0)
'''
