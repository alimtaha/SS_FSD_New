from PIL import Image
from furnace.utils.visualize import print_iou, show_img
from furnace.datasets.BaseDataset import BaseDataset
from furnace.utils.img_utils import generate_random_crop_pos, random_crop_pad_to_shape
from exp_city.city8_res50v3_CPS_CutMix.config import config as conzeft
import random
from torch.utils import data
import numpy as np
import torch
import cv2
import os
import sys

for n in range(1, 4):
    m = '../'
    sys.path.append(m * n)

'''
Functions below called by dataloader for transformations like:
- Normalisation ( transforming image from 0-255 into 0-1, changing mean and std_dev )
- Random mirroring of the image
- Random scaling of the image
- Semantic Edge Detector: Gets the GT image, masks all pixels with a value of 255 and converts to 0, then identifies edges of remaining pixels and thickens them.
        final output is thickened edges of ("wanted") semantic objects

'''


def normalize(img, mean, std, dimg=None, d_mean=None, d_std=None, embeddings=False):
    # pytorch pretrained model need the input range: 0-1
    img = img.astype(np.float32) / 255.0
    img = img - mean
    img = img / std

    if dimg is not None:
        if embeddings == True:
            pass
        else:
            dimg = dimg.astype(np.float32) / 256.0
            dimg = dimg - d_mean
            dimg = dimg / d_std

    return img, dimg


def random_mirror(img, dimg=None, gt=None, embeddings=False):
    if random.random() >= 0.5:
        img = cv2.flip(img, 1)
        if dimg is not None:
            if embeddings == True:
                dimg[0] = dimg[0][:, :, ::-1]
                dimg[1] = dimg[1][:, :, ::-1]
            else:
                dimg = cv2.flip(dimg, 1)
        if gt is not None:
            gt = cv2.flip(gt, 1)

    return img, dimg, gt


def random_scale(img, dimg=None, gt=None, scales=None, embeddings=True):
    scale = random.choice(scales)
    # scale = random.uniform(scales[0], scales[-1])
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)

    print('random scale image size', img.shape)


    if dimg is not None:
        if embeddings == True:
            
            #1/16th image size
            C, H, W = dimg[0].shape
            dimg[0] = torch.from_numpy(np.copy(dimg[0]))
            dimg[0] = torch.nn.functional.interpolate(dimg[0], (C, int(H * scale), int(W * scale)), mode='bilinear').numpy()

            #1/4th image size
            C, H, W = dimg[1].shape


            dimg[1] = torch.from_numpy(np.copy(dimg[1]))
            dimg[1] = torch.nn.functional.interpolate(dimg[1], (C, int(H * scale), int(W * scale)), mode='bilinear').numpy()
            

        else:
            # This may not work, so may need to be changed!!!
            dimg = cv2.resize(dimg, (sw, sh), interpolation=cv2.INTER_NEAREST)

    if gt is not None:
        gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)

    print('after resize', img.shape)

    return img, dimg, gt, scale


def SemanticEdgeDetector(gt):
    # Since only condition is given, a tuple is returned with the indices
    # where the condition is True.
    id255 = np.where(gt == 255)
    no255_gt = np.array(gt)  # converting ground truth image into numpy array
    # all indices where gt has a value of 255 are assigned a new value of 0
    no255_gt[id255] = 0
    # Using Canny edge detector to find image edges 5,5 are the values used
    # for hysteris thresholding,
    cgt = cv2.Canny(no255_gt, 5, 5, apertureSize=7)
    # in this case they are the same so no hysterisis thresholding is
    # provided, instead, all values greater than 5 are an edge and those less
    # are not, apreture size is the size of the Sobel kernel used for the edge
    # detection, Horizontal & Vertical are done independently the combined
    # using sqrt(G_horiz^2 + G_vert^2)
    edge_radius = 7
    # Creates a 7x7 structuring element to be used for dilation below
    edge_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (edge_radius, edge_radius))
    # After edges identified using the edge detector, they are enlarged using
    # the dilation function
    cgt = cv2.dilate(cgt, edge_kernel)
    # print(cgt.max(), cgt.min())
    cgt[cgt > 0] = 1  # Following the dilation, any area with a non zero value is set to one to avoid different intensities of the edges, only interested in binary edges
    return cgt


class TrainPre(
        object):  # This class preprocesses the images using the functions above for TRAINING only
    def __init__(self, img_mean, img_std, dimage_mean=None, dimage_std=None):
        self.img_mean = img_mean
        self.img_std = img_std
        self.dimg_mean = dimage_mean
        self.dimg_std = dimage_std

    def __call__(
            self,
            img,
            dimg=None,
            gt=None,
            unsupervised=False,
            uns_crops=None,
            embeddings=False):
        # gt = gt - 1     # label 0 is invalid, this operation transfers label
        # 0 to label 255
        img, dimg, gt = random_mirror(img, dimg, gt, embeddings=True)
        if conzeft.train_scale_array is not None:
            img, dimg, gt, scale = random_scale(
                img, dimg, gt, conzeft.train_scale_array, embeddings=True)

        # Need to experiment with whether using mean and std dev for the depth
        # images adds value
        img, dimg = normalize(
            img, self.img_mean, self.img_std, dimg, self.dimg_mean, self.dimg_std, embeddings=True)

        if gt is not None:
            cgt = SemanticEdgeDetector(gt)
        else:
            cgt = None

        crop_size = (conzeft.image_height, conzeft.image_width)

        if unsupervised and conzeft.depthmix:
            crop_pos = uns_crops

        else:
            crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)

        p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        if gt is not None:
            # ignore label for image padding changed to 100, all other labels
            # that are 255 are now used for training
            p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, conzeft.ignore_label)
            # ignore label for image padding changed to 100, all other labels
            # that are 255 are now used for training
            p_cgt, _ = random_crop_pad_to_shape(cgt, crop_pos, crop_size, conzeft.ignore_label)
        else:
            p_gt = None
            p_cgt = None

        if dimg is not None:
            p_dimg, _ = random_crop_pad_to_shape(dimg, crop_pos, crop_size, 0, embeddings=True)
        else:
            p_dimg = None

        # colour channel moved to first dimension (from H x W x C to C x H x W)
        p_img = p_img.transpose(2, 0, 1)

        extra_dict = {}

        return p_img, p_dimg, p_gt, p_cgt, extra_dict


class ValPre(object):  # Validation runs pre processing
    def __call__(self, img, gt):
        # gt = gt - 1
        extra_dict = {}
        return img, gt, None, extra_dict


class TrainValPre(object):  # Validation while Training runs pre processing
    def __init__(self, img_mean, img_std, dimage_mean=None, dimage_std=None):
        self.img_mean = img_mean
        self.img_std = img_std
        self.dimg_mean = dimage_mean
        self.dimg_std = dimage_std

    def __call__(self, img, dimg=None, gt=None, embeddings=False):
        img, dimg = normalize(
            img, self.img_mean, self.img_std, dimg, self.dimg_mean, self.dimg_std, embeddings=True)
        img = img.transpose(2, 0, 1)
        # gt = gt - 1
        extra_dict = {}
        return img, dimg, gt, None, extra_dict


'''
get_train_loader creates the training DataLoader - parameters passed in are:
- TrainPre (training images pre-processing)
- data setting dictionary with paths to images and GT
- Dataset object

Returns the training DataLoader object (and train sampler but that's only valid for distributed training)

'''


# Data setting is dictionary with some parameters,
def get_train_loader(
        engine,
        dataset,
        train_source,
        unsupervised=False,
        collate_fn=None,
        pin_memory_flag=True,
        fully_supervised=False):
    data_setting = {'img_root': conzeft.img_root_folder,
                    'gt_root': conzeft.gt_root_folder,
                    'train_source': train_source,
                    'eval_source': conzeft.eval_source}
    train_preprocess = TrainPre(
        conzeft.image_mean,
        conzeft.image_std,
        conzeft.dimage_mean,
        conzeft.dimage_std)

    if unsupervised is False:
        if fully_supervised is False:
            train_dataset = dataset(data_setting, "train", train_preprocess,
                                    conzeft.max_samples, unsupervised=False)
        else:
            train_dataset = dataset(data_setting, "train", train_preprocess,
                                    conzeft.num_train_imgs, unsupervised=False)
    else:
        train_dataset = dataset(data_setting, "train", train_preprocess,
                                conzeft.max_samples, unsupervised=True)

    train_sampler = None
    is_shuffle = True
    batch_size = conzeft.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        batch_size = conzeft.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=conzeft.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=pin_memory_flag,
                                   sampler=train_sampler,
                                   collate_fn=collate_fn)

    return train_loader, train_sampler


class CityScape(BaseDataset):

    # trans_labels = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    #                28, 31, 32, 33]

    trans_labels = [7, 35]  # 35 is random label used as non road label

    def __init__(
            self,
            setting,
            split_name,
            preprocess=None,
            file_length=None,
            training=True,
            unsupervised=False,
            uns_crops=None):
        super(
            CityScape,
            self).__init__(
            setting,
            split_name,
            preprocess,
            file_length)
        self._split_name = split_name
        self._img_path = setting['img_root']
        self._gt_path = setting['gt_root']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length  # file_length here is the conzeft.max_samples
        # Train and Val preprocess classes above, passed in by the DataLoader
        # when it calls the dataset
        self.preprocess = preprocess
        self.training = training
        self.unsupervised = unsupervised
        self.uns_crops = uns_crops          #used to maintain depth mix crops

    def __getitem__(self, index):
        if self._file_length is not None:
            # this bit is inefficient since the filenames are reconstructed
            # everytime the dataloader is called.
            names = self._construct_new_file_names(self._file_length)[index]
        else:
            names = self._file_names[index]
        # - changed to .jpg since images have JPG prefix not PNG, os.path.join(self._img_path, names[0])
        img_path = self._img_path + names[0].split('.')[0] + '.jpg'
        e3_path = self._img_path + '/depth_gen/' + conzeft.depth_ckpt + 'embeddings/' + names[0].split('/')[3] + 'E3.pt'
        e1_path = self._img_path + '/depth_gen/' + conzeft.depth_ckpt + 'embeddings/' + names[0].split('/')[3] + 'E1.pt'
        dpath = (e3_path, e1_path)
        # os.path.join(self._gt_path, names[1])
        if conzeft.weak_labels and self._split_name == 'train':
            names_split = names[1].split('/')
            subdir = names_split[-1].split('_')[0]
            #new_list = [names_split[0:2], ]
            new_names = ('/').join([names_split[1],
                                    names_split[2], subdir, names_split[3]])
            gt_path = (self._gt_path + new_names).replace('/segmentation',
                                                          '/segmentation_weak_0.1').replace('_gtFine', '')
        else:
            gt_path = (self._gt_path + names[1])

        item_name = names[1].split("/")[-1].split(".")[0]

        if not self.unsupervised:
            img, dimg, gt = self._fetch_data(
                img_path, dpath, gt_path, embeddings=True)  # Image opened using cv2.imread
        else:
            img, dimg, gt = self._fetch_data(img_path, dpath, None, embeddings=True)

        img = img[:, :, ::-1]  # flip third dimension

        if self.preprocess is not None:
            if self._split_name == 'train':
                img, dimg, gt, edge_gt, extra_dict = self.preprocess(
                    img, dimg, gt, self.unsupervised, uns_crops=(
                        self.uns_crops[index] if self.unsupervised and conzeft.depthmix else None), embeddings=True)
            else:
                img, dimg, gt, edge_gt, extra_dict = self.preprocess(
                    img, dimg, gt, embeddings=True)
        if gt is not None:
            for i in range(
                    1, 19):  # setting all labels apart from road to 1 (ignore label)
                gt[np.where(gt == i)] = 1
            gt[np.where(gt == 255)] = 1

        # if dimg is not None:
        #     dimg = dimg[np.newaxis, ...]
        #     # Appending depth to last dimension
        #     #img = np.concatenate((img, dimg), axis=0)

        if self._split_name in [
            'train',
            'trainval',
            'train_aug',
            'trainval_aug']:
            # image converted to torch array
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            dimg = (torch.from_numpy(dimg[0].copy()), torch.from_numpy(dimg[1].copy()))
            
            if gt is not None:
                # contiguous: This function returns an array with at least
                # one-dimension (1-d) so it will not preserve 0-d arrays.
                gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
                if self._split_name != 'trainval':
                    # edge_gt is the semantic edge detector function
                    edge_gt = torch.from_numpy(
                        np.ascontiguousarray(edge_gt)).long()

            # converting items in the dictionary to torch tensors (empty dict
            # is returned from the TrainPre function so should not be None)
            if self.preprocess is not None and extra_dict is not None:
                for k, v in extra_dict.items(
                ):  # iterating thru extra_dict key, value pairs
                    extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v))
                    if 'label' in k:
                        extra_dict[k] = extra_dict[k].long()
                    if 'img' in k:
                        extra_dict[k] = extra_dict[k].float()

        output_dict = dict(data=img, fn=str(item_name),  # output_dict initialised with 3 items, the image, the name of the image and the length of the train_loader (file_names)
                           n=len(self._file_names))
        if gt is not None:
            # if a label exists, also appends another value to the dictionary
            extra_dict['label'] = gt

        if dimg is not None:
            # if a label exists, also appends another value to the dictionary
            extra_dict['embeddings'] = dimg        

        if self.preprocess is not None and extra_dict is not None:
            # appending the extra_dict (gt) to output_dict
            output_dict.update(**extra_dict)

        return output_dict

    def _fetch_data(self, img_path, dpath=None, gt_path=None, dtype=None, embeddings=False):
        img = self._open_image(img_path)

        if dpath is not None:
            if embeddings == False:
                dimg = np.array(Image.open(dpath))
            else:
                dimg = [torch.load(dpath[0]), torch.load(dpath[1])]
        else:
            dimg = None

        if gt_path is not None:
            gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype)
            return img, dimg, gt

        return img, dimg, None

    @classmethod
    def get_class_colors(*args):

        # Had to extend to two classes since
        return [[128, 64, 128], [0, 70, 255]]

        '''
        , [244, 35, 232], [70, 70, 70],
                [102, 102, 156], [190, 153, 153], [153, 153, 153],
                [250, 170, 30], [220, 220, 0], [107, 142, 35],
                [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0],
                [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
                [0, 0, 230], [119, 11, 32]]
        '''

    @classmethod
    def get_class_names(*args):
        return ['road', 'not_road']

        '''
        , 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign',
                'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle']
        '''

    @classmethod
    def transform_label(cls, pred, name):
        label = np.zeros(pred.shape)
        # returns only once instance of every label in the image
        ids = np.unique(pred)
        print(ids)  # checking whether the model indeed only returns a a value between 0-18 (19 labels) and they have to be converted to the actual label after through this function
        for id in ids:
            # converting the labels from sequential numbers to the actual
            # labels as per the cityscapes dataset spec
            label[np.where(pred == id)] = cls.trans_labels[id]

        new_name = (name.split('.')[0]).split('_')[:-1]
        new_name = '_'.join(new_name) + '.png'

        return label, new_name
