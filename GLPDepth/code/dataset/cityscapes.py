import os
import cv2
import numpy as np
import torch
from dataset.base_dataset import BaseDataset
import json
from matplotlib import pyplot as plt

# a7a


class cityscapes(BaseDataset):
    def __init__(self, data_path, train_txt, test_txt, filenames_path='./code/dataset/filenames/',
                 is_train=True, dataset='cityscapes', crop_size=(352, 704),
                 scale_size=None, logarithmic=False, disparity=False):
        super().__init__(crop_size)

        self.scale_size = scale_size

        self.is_train = is_train
        self.data_path = data_path

        self.image_path_list = []
        self.depth_path_list = []
        #txt_path = '/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/AdaBins/AdaBins/train_test_inputs/cityscapes_train_extra_edited.txt'

        if is_train:
            txt_path = train_txt #'/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/AdaBins/AdaBins/train_test_inputs/cityscapes_train_extra_edited.txt'
        else:
            txt_path = test_txt #'/home/taha_a@WMGDS.WMG.WARWICK.AC.UK/Documents/ss_fsd/AdaBins/AdaBins/train_test_inputs/cityscapes_val_edited.txt'

        self.logarithmic = logarithmic

        self.disparity = disparity

        print('Logairthmic set to', self.logarithmic)

        if self.disparity:
            print('using disparities')

        self.filenames_list = self.readTXT(txt_path)
        phase = 'train' if is_train else 'test'
        print("Dataset :", dataset)
        print("# of %s images: %d" % (phase, len(self.filenames_list)))

    # kb cropping
    def cropping(self, img, depth):

        h_im, w_im = img.shape[:2]

        margin_h = 1024 - 704
        margin_w = 2048 - 352

        crop_h = np.random.randint(0, margin_h)
        crop_w = np.random.randint(0, margin_w)

        img = img[crop_h: crop_h + 704,
                  crop_w: crop_w + 352]

        depth = depth[crop_h: crop_h + 704,
                      crop_w: crop_w + 352]

        return img, depth

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx): 

        img_path = os.path.join(self.data_path,'images/images/') + self.filenames_list[idx].split(' ')[0]

        if self.disparity:
            gt_path = os.path.join(self.data_path,'depth/disparity/') + \
                self.filenames_list[idx].split(' ')[1].replace('.png', '_disparity.png')
            camera_path = os.path.join(self.data_path,'camera') + \
                self.filenames_list[idx].split(' ')[2]
        else:
            gt_path = os.path.join(self.data_path,'depth/depth256/') + \
                self.filenames_list[idx].split(' ')[1]

        filename = img_path.split('/')[-1]  # + '_' + img_path.split('/')[-1]

        image = cv2.imread(img_path)  # [H x W x C] and C: BGR
        # check whether this is needed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype('float32')

        if self.disparity:  # load camera parameters and convert to metres
            with open(camera_path, 'r') as f:
                camera_params = json.loads(f.read())
                depth[depth > 0] = (depth[depth > 0] - 1) / 256
                depth[depth > 0] = camera_params['extrinsic']['baseline'] * (
                    (camera_params['intrinsic']['fx'] + camera_params['intrinsic']['fy']) / 2) / depth[depth > 0]
                depth[depth == np.inf] = 0
                depth[depth == np.nan] = 0
        else:
            depth = depth / 256.0  # convert in meters

        # image, depth = self.cropping(image, depth) - this is for kb cropping,
        # not needed for cityscapes, cropping done in 'augment_training_data'
        # below

        if self.scale_size:
            image = cv2.resize(image, (self.scale_size[0], self.scale_size[1]))
            depth = cv2.resize(depth, (self.scale_size[0], self.scale_size[1]))

        if self.is_train:
            image, depth = self.augment_training_data(image, depth)
        else:
            image, depth = self.augment_test_data(image, depth)

        depth_clone = depth.clone()
        test = False

        if self.logarithmic and self.is_train:
            print('log')
            depth = 0.25 * (torch.log(depth + (1.01**200.0)) /
                            torch.log(torch.tensor(1.01))) - 50

        if test:
            depth[depth > 256] = 0
            fig = plt.figure(figsize=(10, 7))
            fig.add_subplot(1, 3, 1)
            plt.imshow(image.permute(1, 2, 0))
            fig.add_subplot(1, 3, 2)
            plt.imshow(depth, cmap='magma_r', vmin=0, vmax=256)
            plt.colorbar()
            fig.add_subplot(1, 3, 3)
            plt.imshow(depth_clone, cmap='magma_r', vmin=0, vmax=256)
            plt.colorbar()
            plt.show()

        # convert to logarithmic depth to aid with large depth values - however
        # may lose precision in near field depth values....

        return {'image': image, 'depth': depth, 'filename': filename}
