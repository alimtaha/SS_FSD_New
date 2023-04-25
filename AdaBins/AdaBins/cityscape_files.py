import os
import shutil


trainpath = '/home/extraspace/Datasets/Datasets/cityscapes/city/images/train'
valpath = '/home/extraspace/Datasets/Datasets/cityscapes/city/images/val'
train_files_semiseg = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/AdaBins/AdaBins/train_files_semiseg'
val_files_semiseg = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/AdaBins/AdaBins/val_files_semiseg'
count = 0

with open(val_files_semiseg, 'w') as vfile:
    for f in os.listdir(valpath):
        vfile.write(f + ' None' + '\n')
    # print(f)
        #print(os.path.isfile(os.path.join(mypath, f)))
        #rename = f.split('_')[2:6]
        #name = '_'.join(rename)
        #os.rename(os.path.join(trainpath, f), os.path.join(trainpath, name))
        # tfile.write(name)
        # tfile.write('\n')
