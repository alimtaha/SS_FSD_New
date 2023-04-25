import random

fp_shuffle = '/vol/biomedic2/aa16914/shared/amir_ali/datasets/config_new/config_new/subset_train/train_aug_unlabeled_1-16_shuffle.txt'
fp = '/vol/biomedic2/aa16914/shared/amir_ali/datasets/config_new/config_new/subset_train/train_aug_unlabeled_1-16.txt'

with open(fp, 'r') as f:
    unlabelled = list(f.readlines())    
    random.shuffle(unlabelled)

with open(fp_shuffle, 'w') as fw:
    fw.writelines(unlabelled)

print('done!')