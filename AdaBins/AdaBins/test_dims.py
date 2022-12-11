from PIL import Image
import os
import numpy as np

path = '/home/extraspace/Datasets/cityscapes/Depth_Training_Extra/leftImg8bit'
num = 0

with open('/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/AdaBins/AdaBins/train_test_inputs/cityscapes_train_extra_edited.txt', 'r') as r:
    for line in r:
        line = line.split()[0]
        p = os.path.join(path, line)
        size = len(np.array(Image.open(p)).shape)
        if size == 3:
            num += 1
        else:
            print(p, size)

    print('total number', num)
