import os
import shutil


files = ['dummy']

with open('AdaBins/train_test_inputs/kitti_eigen_test_files_with_gt.txt', 'r') as test:
    for line in test:
        line = line.split('/')[1:2]
        line = ('/').join(line)
        line = ('_').join(line.split('_')[0:5])
        for x in files:
            if x == line:
                break
            elif x == files[-1]:
                files.append(line)

files.pop(0)
files.sort()

with open('AdaBins/train_test_inputs/kitti_test_downloads', 'w') as tst:
    for x in files:
        tst.write(x)
        tst.write('\n')


# second approaching by deleting duplicates
'''
files = []

with open('AdaBins/train_test_inputs/kitti_eigen_train_files_with_gt.txt', 'r') as train:
    for line in train:
        line = line.split('/')[0:2]
        line = ('/').join(line)
        files.append(line)

a = list(set(files))
a.sort()
print(len(a))
'''
