import os
import shutil


# files with all semiseg  file names (to feed below files)
trainpath = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/CPS/TorchSemiSeg/DATA/city/config_new/train.txt'
valpath = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/CPS/TorchSemiSeg/DATA/city/config_new/val.txt'
testpath = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/CPS/TorchSemiSeg/DATA/city/config_new/test.txt'

# files to be written to with all semiseg training file names (used for
# running MDE)
trainwrite = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/CPS/TorchSemiSeg/Depth_File_Names/trainfiles depth_disparity.txt'
valwrite = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/CPS/TorchSemiSeg/Depth_File_Names/valfiles depth_disparity.txt'
testwrite = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/CPS/TorchSemiSeg/Depth_File_Names/testfiles depth_disparity.txt'


file_list = [trainpath, valpath, testpath]
filewrite_list = [trainwrite, valwrite, testwrite]

images_path = '/home/extraspace/Datasets/Datasets/cityscapes/leftImg8bit_trainvaltest (blurred)/leftImg8bit'

# Renaming blurred images to match convention
'''
for m in os.listdir(images_path):
    m_path = os.path.join(images_path,m)
    for n in os.listdir(m_path):
        n_path = os.path.join(m_path,n)
        for o in os.listdir(n_path):
            pth = os.path.join(n_path,o)
            rename = ('_').join(o.split('_')[0:3])+'.jpg'
            print(rename)
            os.rename(pth, os.path.join(n_path, rename))

'''

# Filenames used during creating depth maps (MDE inference) which will
# then be used during seg training run
for filename, filewrite in zip(file_list, filewrite_list):
    with open(filewrite, 'w') as file:
        for f in open(filename, 'r'):
            #print(os.path.isfile(os.path.join(mypath, f)))
            rename = f.split()[0].split('/')[2:]  # /train.....
            print(len(rename))
            city = rename[1].split('_')[0]  # jena
            mode = rename[0]  # train
            name = '/' + mode + '/' + city + '/' + \
                rename[1].split('.')[0] + '.jpg'
            file.write(name)
            file.write('\n')
