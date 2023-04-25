import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image
import os
import numpy as np


def rot_y(angle):
    cost = np.cos(np.deg2rad(angle))
    sint = np.sin(np.deg2rad(angle))
    rot = np.array([[cost, 0, sint],
                    [0, 1, 0],
                    [-sint, 0, cost]])
    return rot


def rot_z(angle):
    cost = np.cos(np.deg2rad(angle))
    sint = np.sin(np.deg2rad(angle))
    rot = np.array([[cost, -sint, 0],
                    [sint, cost, 0],
                    [0, 0, 1]])
    return rot


def rot_x(angle):
    cost = np.cos(np.deg2rad(angle))
    sint = np.sin(np.deg2rad(angle))
    rot = np.array([[1, 0, 0],
                    [0, cost, -sint],
                    [0, sint, cost]])
    return rot


input_pts = np.float32([[0, 0], [0, 1024], [2048, 1024], [2048, 0]])
output_pts = np.float32([[600, 345], [160, 900], [2000, 900], [1250, 345]])

input_pts = np.float32([[920, 547], [528, 803], [1895, 803], [1329, 547]])
output_pts = np.float32([[0, 0], [0, 900], [1800, 900], [1800, 0]])

# M=rot_z(0)@rot_y(0)@rot_x(1)

img = np.array(Image.open(
    '/home/extraspace/Datasets/cityscapes/Depth_Training_Extra/leftImg8bit/train_extra/mannheim/mannheim_000000_001126.jpg'))

plt.imshow(img)
plt.show()

#img = cv.imopen()

M = cv.getPerspectiveTransform(input_pts, output_pts)

#M=cv.getRotationMatrix2D(center=(0,0), angle=90, scale=1)
print(M)


imp = cv.warpPerspective(img, M, (2048, 1024))

plt.imshow(imp)
plt.show()
