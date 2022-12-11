import numpy as np
import cv2
from PIL import Image
import os
from matplotlib import pyplot as plt

label = '/home/extraspace/Datasets/Datasets/cityscapes/city/segmentation/val/frankfurt_000000_000294_gtFine.png'

label = np.array(Image.open(label), dtype=np.int)

mask = np.array(label > 250).astype(int)
mask[:600,...] = 0

plt.imshow(mask)
plt.show()

