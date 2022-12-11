import copy
import math
from nis import cat
import os
import os.path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from matplotlib import pyplot as plt
from PIL import Image

from . import augmentation as psp_trsform
from .base import BaseDataset
from .sampler import DistributedGivenIterationSampler


class city_dset(BaseDataset):
    def __init__(
            self,
            data_root,
            data_list,
            trs_form,
            seed,
            n_sup,
            split="val",
            depth_root=None,
            unsupervised=False):
        super(city_dset, self).__init__(data_list)
        self.data_root = data_root
        self.depth_root = depth_root
        self.transform = trs_form
        self.uns_crops = None
        self.unsupervised = unsupervised
        random.seed(seed)
        if len(self.list_sample) >= n_sup and split == "train":
            self.list_sample_new = random.sample(self.list_sample, n_sup)
        elif len(self.list_sample) < n_sup and split == "train":
            num_repeat = math.ceil(n_sup / len(self.list_sample))
            self.list_sample = self.list_sample * num_repeat

            self.list_sample_new = random.sample(self.list_sample, n_sup)
        else:
            self.list_sample_new = self.list_sample

        print('number of samples - Cityscapes,py', len(self.list_sample_new))

    def img_loader(self, path, mode, depth=False):
        # with open(path, "rb") as f:
        img = Image.open(path)
        if depth is False:
            return img.convert(mode)
        else:
            return img

    def __getitem__(self, index):
        # load image and its label
        image_path = os.path.join(
            self.data_root,
            self.list_sample_new[index][0])
        label_path = os.path.join(
            self.data_root,
            self.list_sample_new[index][1])
        #print(image_path, label_path)
        # image = self.img_loader('/home/extraspace/Datasets/Datasets/cityscapes/city/images/train/darmstadt_000014_000019.jpg', 'RGB') #image_path, "RGB")
        # label =
        # self.img_loader('/home/extraspace/Datasets/Datasets/cityscapes/city/segmentation/train/darmstadt_000014_000019_gtFine.png',
        # 'L') #label_path, "L")
        image = self.img_loader(image_path, "RGB")
        label = self.img_loader(label_path, "L")
        if self.unsupervised:
            crops = self.uns_crops[index]
        else:
            crops = None
        if self.depth_root is not None:
            depth_subpath = self.list_sample_new[index][0].split(
                '/')[2].replace('.jpg', '.png')
            depth_path = os.path.join(self.depth_root, depth_subpath)
            depth_image = self.img_loader(depth_path, "L", depth=True)
            image, label, depth_image = self.transform(
                image, label, depth_image, crops)  # add depth image back in'''
            image = torch.cat((image, depth_image), dim=1)
        else:
            image, label, _ = self.transform(image, label, None, crops)
        for i in range(1, 19):
            label[0, 0][np.where(label[0, 0] == i)] = 1
        label[0, 0][np.where(label[0, 0] == 255)] = 1
        # change all non road labels to 1:

        name = self.list_sample_new[index][0]

        return image[0], label[0, 0].long(), name

    def __len__(self):
        return len(self.list_sample_new)


def build_transfrom(cfg, depth=False, uns=False):
    trs_form = []
    if depth:
        mean, std, ignore_label, dmean, dstd = cfg["mean"], cfg[
            "std"], cfg["ignore_label"], cfg["dmean"], cfg["dstd"]
    else:
        mean, std, ignore_label = cfg["mean"], cfg["std"], cfg["ignore_label"]
        dmean = dstd = None
    trs_form.append(psp_trsform.ToTensor())
    trs_form.append(
        psp_trsform.Normalize(
            mean=mean,
            std=std,
            dmean=dmean,
            dstd=dstd))
    if cfg.get("resize", False):  # .get looks for the key in the dicionary, if it exists is loads the value for that key otherwise it outputs the second value in the function call/argument (set as False ehre)
        trs_form.append(psp_trsform.Resize(cfg["resize"]))
    if cfg.get("rand_resize", False):
        trs_form.append(psp_trsform.RandResize(cfg["rand_resize"]))
    if cfg.get("rand_rotation", False):
        rand_rotation = cfg["rand_rotation"]
        trs_form.append(
            psp_trsform.RandRotate(rand_rotation, ignore_label=ignore_label)
        )
    if cfg.get("GaussianBlur", False) and cfg["GaussianBlur"]:
        trs_form.append(psp_trsform.RandomGaussianBlur())
    if cfg.get("flip", False) and cfg.get("flip"):
        trs_form.append(psp_trsform.RandomHorizontalFlip())
    if cfg.get("crop", False):
        crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
        trs_form.append(
            psp_trsform.Crop(
                crop_size,
                crop_type=crop_type,
                ignore_label=ignore_label))
    if cfg.get("cutout", False):
        n_holes, length = cfg["cutout"]["n_holes"], cfg["cutout"]["length"]
        trs_form.append(psp_trsform.Cutout(n_holes=n_holes, length=length))
    if cfg.get("cutmix", False):
        n_holes, prop_range = cfg["cutmix"]["n_holes"], cfg["cutmix"]["prop_range"]
        trs_form.append(
            psp_trsform.Cutmix(
                prop_range=prop_range,
                n_holes=n_holes))

    return psp_trsform.Compose(trs_form)


def build_cityloader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]
    cfg_trainer = all_cfg["trainer"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 2975)
    print('in cityscapes.py - number of sup', n_sup)

    # build transform
    trs_form = build_transfrom(cfg)
    dset = city_dset(
        cfg["data_root"],
        cfg["data_list"],
        trs_form,
        seed,
        n_sup,
        split)

    # build sampler
    #sample = DistributedSampler(dset)
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        # sampler=sample,
        shuffle=False,
        pin_memory=False,
    )
    return loader


def build_cityloader_depth(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]
    cfg_trainer = all_cfg["trainer"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 2975)
    print(cfg)
    print(cfg["train"]["depth_root"])

    # build transform
    trs_form = build_transfrom(cfg, depth=True)
    dset = city_dset(
        cfg["data_root"],
        cfg["data_list"],
        trs_form,
        seed,
        n_sup,
        split,
        depth_root=cfg["train"]["depth_root"])

    # build sampler
    #sample = DistributedSampler(dset)
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        # sampler=sample,
        shuffle=False,
        pin_memory=False,
    )
    return loader


def build_city_semi_loader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = 2975 - cfg.get("n_sup", 2975)

    # build transform
    trs_form = build_transfrom(cfg)
    trs_form_unsup = build_transfrom(cfg)
    dset = city_dset(
        cfg["data_root"],
        cfg["data_list"],
        trs_form,
        seed,
        n_sup,
        split)

    if split == "val":
        # build sampler
        #sample = DistributedSampler(dset)
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            # sampler=sample,
            shuffle=False,
            pin_memory=True,
        )
        return loader

    else:
        # build sampler for unlabeled set
        data_list_unsup = cfg["data_list"].replace(
            "labeled.txt", "unlabeled.txt")
        dset_unsup = city_dset(
            cfg["data_root"],
            cfg["unlabeled_list"],
            trs_form_unsup,
            seed,
            n_sup,
            split,
            None,
            True)

        dset_unsup_aug = city_dset(
            cfg["data_root"],
            cfg["unlabeled_list_randomized"],
            trs_form_unsup,
            seed,
            n_sup,
            split,
            None,
            True)

        #sample_sup = DistributedSampler(dset)
        loader_sup = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            # sampler=sample_sup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        #sample_unsup = DistributedSampler(dset_unsup)
        loader_unsup = DataLoader(
            dset_unsup,
            batch_size=batch_size,
            num_workers=workers,
            # sampler=sample_unsup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        loader_unsup_aug = DataLoader(
            dset_unsup_aug,
            batch_size=batch_size,
            num_workers=workers,
            # sampler=sample_unsup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        return loader_sup, loader_unsup, loader_unsup_aug


def build_city_semi_depth_loader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = 2975 - cfg.get("n_sup", 2975)
    depth = True

    # build transform
    trs_form = build_transfrom(cfg, depth=depth)
    trs_form_unsup = build_transfrom(cfg, depth=depth)
    dset = city_dset(
        cfg["data_root"],
        cfg["data_list"],
        trs_form,
        seed,
        n_sup,
        split,
        depth_root=cfg["depth_root"])

    if split == "val":
        # build sampler
        #sample = DistributedSampler(dset)
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            # sampler=sample,
            shuffle=False,
            pin_memory=True,
        )
        return loader

    else:
        # build sampler for unlabeled set
        #crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
        data_list_unsup = cfg["data_list"].replace(
            "labeled.txt", "unlabeled.txt")
        #trs_form_unsup = psp_trsform.Compose([psp_trsform.ToTensor(), psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=100)])

        dset_unsup = city_dset(
            cfg["data_root"],
            cfg["unlabeled_list"],
            trs_form_unsup,
            seed,
            n_sup,
            split,
            depth_root=cfg["depth_root"],
            unsupervised=True)

        dset_unsup_aug = city_dset(
            cfg["data_root"],
            cfg["unlabeled_list_randomized"],
            trs_form_unsup,
            seed,
            n_sup,
            split,
            depth_root=cfg["depth_root"],
            unsupervised=True)

        #sample_sup = DistributedSampler(dset)
        loader_sup = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            # sampler=sample_sup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        #sample_unsup = DistributedSampler(dset_unsup)
        loader_unsup = DataLoader(
            dset_unsup,
            batch_size=batch_size,
            num_workers=workers,
            # sampler=sample_unsup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        loader_unsup_aug = DataLoader(
            dset_unsup_aug,
            batch_size=batch_size,
            num_workers=workers,
            # sampler=sample_unsup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        return loader_sup, loader_unsup, loader_unsup_aug
