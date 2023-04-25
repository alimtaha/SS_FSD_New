import path
import os

with open('/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/MDT/Monocular-Depth-Estimation-Toolbox/splits/cityscapes_train_extra_edited.txt', 'w') as tf:
    for m in open(
        '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/MDT/Monocular-Depth-Estimation-Toolbox/splits/cityscapes_train_extra.txt',
            'r'):
        splt = m.split()
        splt[0] = splt[0].replace("_leftImg8bit.png", ".jpg")
        splt[1] = splt[1].replace("_disparity", "")
        line = (" ").join(splt)
        tf.write(line)
        tf.write('\n')

with open('/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/MDT/Monocular-Depth-Estimation-Toolbox/splits/cityscapes_val_edited.txt', 'w') as tf:
    for m in open(
        '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/MDT/Monocular-Depth-Estimation-Toolbox/splits/cityscapes_val.txt',
            'r'):
        splt = m.split()
        splt[0] = splt[0].replace("_leftImg8bit.png", ".jpg")
        splt[0] = splt[0].replace("leftImg8bit/", "")
        splt[1] = splt[1].replace("_disparity", "")
        splt[1] = splt[1].replace("disparity/", "")
        line = (" ").join(splt)
        tf.write(line)
        tf.write('\n')
