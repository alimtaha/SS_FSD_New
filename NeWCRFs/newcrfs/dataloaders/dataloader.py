import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from torchvision import transforms

import numpy as np
from PIL import Image
import os, sys
from matplotlib import pyplot as plt
import random
import json
sys.path.append('../')
sys.path.append('../../')

from utils import DistributedSamplerNoEvenlyDivisible


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode, args):
    return transforms.Compose([
        ToTensor(mode=mode, args=args)
    ])


class NewDataLoader(object):
    def __init__(self, args, mode, world_size=1, rank=0, shuffle=False, drop_last=False):
        if mode == 'train':
            self.training_samples = DataLoadPreprocess(
                args, mode, transform=preprocessing_transforms(mode, args))
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.training_samples, num_replicas=world_size, rank=rank, 
                                shuffle=shuffle, drop_last=drop_last)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=args.num_threads,
                                   pin_memory=False,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            self.testing_samples = DataLoadPreprocess(
                args, mode, transform=preprocessing_transforms(mode, args))
            #if args.distributed:
                # self.eval_sampler = torch.utils.data.distributed.DistributedSampler(self.testing_samples, shuffle=False)
                #self.eval_sampler = DistributedSamplerNoEvenlyDivisible(
                    #self.testing_samples, shuffle=False)
            #else:
            self.eval_sampler = None
            self.data = DataLoader(self.testing_samples, 1,
                                   shuffle=False,
                                   num_workers=8,
                                   pin_memory=True,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            self.testing_samples = DataLoadPreprocess(
                args, mode, transform=preprocessing_transforms(mode, args))
            self.data = DataLoader(
                self.testing_samples,
                1,
                shuffle=False,
                num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))

class DataLoadPreprocess(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False):
        self.args = args
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        # focal = float(sample_path.split()[2])
        focal = 518.8579

        if self.mode == 'train':
            if self.args.dataset == 'kitti':
                rgb_file = sample_path.split()[0]
                depth_file = os.path.join(
                    sample_path.split()[0].split('/')[0],
                    sample_path.split()[1])
                if self.args.use_right is True and random.random() > 0.5:
                    rgb_file.replace('image_02', 'image_03')
                    depth_file.replace('image_02', 'image_03')
            else:
                rgb_file = sample_path.split()[0]
                depth_file = sample_path.split()[1]
                camera_file = sample_path.split()[2]

            image_path = os.path.join(self.args.data_path, rgb_file)
            filename = image_path.split('/')[-1].replace('.jpg', '.png')

            image = Image.open(image_path)
            
            if self.args.disparity:
                depth_path = os.path.join(self.args.disparity_path, depth_file).replace('.png', '_disparity.png')
                camera_path = os.path.join(self.args.camera_path, camera_file)
            else:
                depth_path = os.path.join(self.args.gt_path, depth_file)

            depth_gt = Image.open(depth_path)

            if self.args.do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352))
                image = image.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352))

            # To avoid blank boundaries due to pixel registration
            if self.args.dataset == 'nyu':
                if self.args.input_height == 480:
                    depth_gt = np.array(depth_gt)
                    valid_mask = np.zeros_like(depth_gt)
                    valid_mask[45:472, 43:608] = 1
                    depth_gt[valid_mask == 0] = 0
                    depth_gt = Image.fromarray(depth_gt)
                else:
                    depth_gt = depth_gt.crop((43, 45, 608, 472))
                    image = image.crop((43, 45, 608, 472))

            if self.args.dataset == 'cityscapes':
                height = image.height
                width = image.width
                top_margin = int(height - 1024)
                left_margin = int((width - 2048) / 2)
                depth_gt = depth_gt.crop(
                    (left_margin, top_margin, left_margin + 2048, top_margin + 1024))
                image = image.crop(
                    (left_margin, top_margin, left_margin + 2048, top_margin + 1024))
                
            if self.args.do_random_rotate is True:
                random_angle = (random.random() - 0.5) * 2 * self.args.degree
                image = self.rotate_image(image, random_angle)
                depth_gt = self.rotate_image(depth_gt, random_angle, flag=Image.NEAREST)

            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)

            if self.args.dataset == 'nyu':
                depth_gt = depth_gt / 1000.0
            else:
                if self.args.disparity:
                    with open(camera_path, 'r') as f:   # load camera parameters and convert to metres
                        camera_params = json.loads(f.read())
                        depth_gt[depth_gt > 0] = (depth_gt[depth_gt > 0] - 1) / 256
                        depth_gt[depth_gt > 0] = camera_params['extrinsic']['baseline'] * (
                            (camera_params['intrinsic']['fx'] + camera_params['intrinsic']['fy']) / 2) / depth_gt[depth_gt > 0]
                        depth_gt[depth_gt == np.inf] = 0
                        depth_gt[depth_gt == np.nan] = 0 
                else:
                    depth_gt = depth_gt / 256.0

            if image.shape[0] != self.args.input_height or image.shape[1] != self.args.input_width:
                image, depth_gt = self.random_crop(image, depth_gt, self.args.input_height, self.args.input_width)

            image, depth_gt = self.train_preprocess(image, depth_gt)
            sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'filename': filename}

        else:
            if self.mode == 'online_eval':
                data_path = self.args.data_path_eval
            else:
                data_path = self.args.data_path

            image_path = os.path.join(data_path, "./" + sample_path.split()[0])
            image = np.asarray(Image.open(image_path),dtype=np.float32) / 255.0

            filename = image_path.split('/')[-1].replace('.jpg', '.png')
            camera_file = sample_path.split()[2]

            if self.mode == 'online_eval':
                gt_path = self.args.gt_path_eval
                if self.args.disparity:
                    depth_path = os.path.join(
                    self.args.disparity_path, "./" + sample_path.split()[1]).replace('.png', '_disparity.png')
                    camera_path = os.path.join(self.args.camera_path, camera_file)
                else:
                    depth_path = os.path.join(
                    gt_path, "./" + sample_path.split()[1])
                if self.args.road_mask:
                    semantic_path = os.path.join(self.args.semantic_labels_dataset, sample_path.split()[1].split('/')[-1].replace('.png', '_gtFine.png'))

                if self.args.dataset == 'kitti':
                    depth_path = os.path.join(
                        gt_path,
                        sample_path.split()[0].split('/')[0],
                        sample_path.split()[1])
                has_valid_depth = False
                try:
                    depth_gt = Image.open(depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    if self.args.dataset == 'nyu':
                        depth_gt = depth_gt / 1000.0
                    else:
                        if self.args.disparity:
                            with open(camera_path, 'r') as f:   # load camera parameters and convert to metres
                                camera_params = json.loads(f.read())
                                depth_gt[depth_gt > 0] = (depth_gt[depth_gt > 0] - 1) / 256
                                depth_gt[depth_gt > 0] = camera_params['extrinsic']['baseline'] * (
                                    (camera_params['intrinsic']['fx'] + camera_params['intrinsic']['fy']) / 2) / depth_gt[depth_gt > 0]
                                depth_gt[depth_gt == np.inf] = 0
                                depth_gt[depth_gt == np.nan] = 0 
                        else:
                            depth_gt = depth_gt / 256.0

                    if self.args.road_mask:
                        road_mask = Image.open(semantic_path)
                        road_mask = np.asarray(road_mask)
                        road_mask  = np.where(road_mask == 0, 1, 0)
                        road_mask = np.expand_dims(road_mask, axis=2)

            if self.args.do_kb_crop is True:
                height = image.shape[0]
                width = image.shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                image = image[top_margin:top_margin + 352,
                              left_margin:left_margin + 1216, :]
                if self.mode == 'online_eval' and has_valid_depth:
                    depth_gt = depth_gt[top_margin:top_margin +
                                        352, left_margin:left_margin + 1216, :]

            if self.mode == 'online_eval':
                if self.args.road_mask:
                    sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth, 'path': image_path, 'filename': filename, 'road_mask': road_mask}
                else:
                    sample = {'image': image, 'depth': depth_gt, 'focal': focal, 'has_valid_depth': has_valid_depth, 'path': image_path, 'filename': filename}
            else:
                sample = {'image': image, 'focal': focal, 'filename': filename}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def rotate_image(self, image, angle, flag=Image.BILINEAR):
        result = image.rotate(angle, resample=flag)
        return result

    def random_crop(self, img, depth, height, width):
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == depth.shape[0]
        assert img.shape[1] == depth.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth

    def train_preprocess(self, image, depth_gt):
        # Random flipping
        do_flip = random.random()
        if do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()

        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)

        return image, depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.args.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug

    def __len__(self):
        return len(self.filenames)


class ToTensor(object):
    def __init__(self, mode, args):
        self.mode = mode
        self.args = args
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, focal, filename = sample['image'], sample['focal'], sample['filename']
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == 'test':
            return {'image': image, 'focal': focal, 'filename': filename}

        depth = sample['depth']
        if self.args.road_mask:
            road_mask = sample['road_mask']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            if self.args.road_mask:
                road_mask = self.to_tensor(road_mask)
            return {'image': image, 'depth': depth, 'focal': focal, 'filename': filename}
        else:
            has_valid_depth = sample['has_valid_depth']
            if self.args.road_mask:
                return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth, 'path': sample['path'], 'filename': filename, 'road_mask': road_mask}
            else:
                return {'image': image, 'depth': depth, 'focal': focal, 'has_valid_depth': has_valid_depth, 'path': sample['path'], 'filename': filename}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
