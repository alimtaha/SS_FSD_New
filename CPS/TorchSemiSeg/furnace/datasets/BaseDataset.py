#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2017/12/16 下午8:41
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : BaseDataset.py

import os
import time
import cv2
import torch
import numpy as np

import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None,
                 file_length=None):
        super(BaseDataset, self).__init__()
        self._split_name = split_name
        self._img_path = setting['img_root']
        self._gt_path = setting['gt_root']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            names = self._construct_new_file_names(self._file_length)[index]
        else:
            # names takes the required index from the filenames, depending on
            # the file_length variable a new file_names file with a custom
            # length may first be constructed before getting an index
            names = self._file_names[index]
        img_path = os.path.join(self._img_path, names[0])
        gt_path = os.path.join(self._gt_path, names[1])
        # getting the actual name of the file (not path) of the sample, so for
        # example 'jena_000114_000019_gtFine
        item_name = names[1].split("/")[-1].split(".")[0]

        img, gt = self._fetch_data(img_path, gt_path)

        img = img[:, :, ::-1]  # flip the channels?
        if self.preprocess is not None:
            img, gt, extra_dict = self.preprocess(img, gt)

        if self._split_name is 'train':
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            if self.preprocess is not None and extra_dict is not None:
                for k, v in extra_dict.items():
                    extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v))
                    if 'label' in k:
                        extra_dict[k] = extra_dict[k].long()
                    if 'img' in k:
                        extra_dict[k] = extra_dict[k].float()

        output_dict = dict(data=img, label=gt, fn=str(item_name),
                           n=len(self._file_names))
        if self.preprocess is not None and extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _fetch_data(self, img_path, gt_path, dtype=None):
        img = self._open_image(img_path)
        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype)

        return img, gt

    def _get_file_names(self, split_name, train_extra=False):
        assert split_name in ['train', 'val', 'trainval']
        source = self._train_source
        if split_name == 'val' or split_name == 'trainval':
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            img_name, gt_name = self._process_item_names(item)
            file_names.append([img_name, gt_name])

        if train_extra:
            file_names2 = []
            source2 = self._train_source.replace('train', 'train_extra')
            with open(source2) as f:
                files2 = f.readlines()

            for item in files2:
                img_name, gt_name = self._process_item_names(item)
                file_names2.append([img_name, gt_name])

            return file_names, file_names2

        return file_names

    # max samples depending on whether the labelled or unlabelled images are
    # greater in number
    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        # 原来一轮迭代的长度 - The length of the original iteration
        files_len = len(self._file_names)

        # 仅使用小部分数据 - Use only a small portion of the data
        if length < files_len:
            return self._file_names[:length]

        # Author - The length of one iteration obtained according to the
        # setting (按照设定获得的一轮迭代的长度)
        new_file_names = self._file_names * (length // files_len)

        # returns random order of indexes from 0 to n-1 (where n here is the
        # files_len variable)
        rand_indices = torch.randperm(files_len).tolist()
        # if length is greater than files_len, we want to add to the
        # new_file_names list, by augmenting random indices hence the +=
        # statement at the end, only wants length-files_len indices
        new_indices = rand_indices[:length % files_len]

        new_file_names += [self._file_names[i] for i in new_indices]

        return new_file_names

        # the above functionality is needed because of the nature of the training. For every batch we're feeding in an equal number of labelled and unlabelled
        # but obviously due to the labelled ratio this means the unsup and supervised data loaders will have different lengths. Since we need them to be equal, we artificially increase the size of the smaller dataset to match
        # to explain, if supervised samples are 200 and unsupervised are 700, length will be 700 so the new filenames will first multiply the supervised filenames by length // files_len which is 3, so 200*3 is 600. Still an outstanding 100 so
        # length (700) % files_len (200) will give 100, we take 100 more random
        # indices from the dataset and add - result is filenames file with same
        # length as the unsupervised file

    def mask_images(self, img, pixel_factor, masking_size, color_channel=False):

        if torch.is_tensor(img) is True:
            img = img.numpy()
            img = img.transpose(1,2,0)

        h, w, c = img.shape
        h_reduced = h // pixel_factor
        w_reduced = w // pixel_factor

        if color_channel is False:
            c = 1

        masks = np.ones(h_reduced * w_reduced * c)
        indices = np.arange(0,h_reduced * w_reduced * c)
        indices = np.random.choice(indices, size= c * h_reduced * w_reduced // masking_size, replace=False)

        masks[indices] = 0
        masks = masks.reshape(h_reduced, w_reduced, c).astype(np.int)
        masks = cv2.resize(masks, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

        if color_channel is False:
            masks = masks[...,np.newaxis]

        return masks


    def downsample_blur(self, img, downsample_factor, blur=False, kernel=None):
        
        if torch.is_tensor(img) is True:
            img = img.numpy()
            img = img.transpose(1,2,0)

        h, w, _ = img.shape
        h_reduced = h // downsample_factor
        w_reduced = w // downsample_factor

        im_down = cv2.resize(img, dsize=(w_reduced, h_reduced))
        im_down = cv2.resize(im_down, dsize=(w, h))

        if blur:
            im_down = cv2.GaussianBlur(im_down, tuple(kernel), 0)
        
        return im_down



    @staticmethod
    def _process_item_names(item):
        item = item.strip()
        item = item.split('\t')
        img_name = item[0]

        if len(item) == 1:
            gt_name = None
        else:
            gt_name = item[1]

        return img_name, gt_name

    def get_length(self):
        return self.__len__()

    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        # cv2: B G R
        # h w c
        img = np.array(cv2.imread(filepath, mode), dtype=dtype)

        return img

    @classmethod
    def get_class_colors(*args):
        raise NotImplementedError

    @classmethod
    def get_class_names(*args):
        raise NotImplementedError


if __name__ == "__main__":
    data_setting = {'img_root': '',
                    'gt_root': '',
                    'train_source': '',
                    'eval_source': ''}
    bd = BaseDataset(data_setting, 'train', None)
    print(bd.get_class_names())
